"""EQ1 'monitor invocation cost' panel: single-invocation latency of ONE monitor on recorded views.

    python -m evals_runner.eq1_monitor_panel --monitor gemma4-e4b --url http://127.0.0.1:8100 \\
        --prompts eq1_monitor_panel_prompts.json.gz --manifest eq1_monitor_panel_prompts.manifest.json \\
        --model-dir <HF_CACHE>/models--google--gemma-4-E4B-it/snapshots/<rev> \\
        --server-log server.log --server-info server_info.json --out-dir out/gemma4-e4b
    python -m evals_runner.eq1_monitor_panel --combine out            # one table over every monitor directory in out/
    python -m evals_runner.eq1_monitor_panel --not-measured NAME --reason "..." --out-dir out/NAME

What is measured. The registered judge (registry name) reads the user messages in the prompt file (built by
eq1_monitor_panel_prompts.py from recorded training-partition trajectories; no agent is run) one request at a time
(concurrency 1), under the registry's request body, reasoning switch, sampling, seed and output cap, with natural stops (no
forced length). The output cap and the reply parsing are those of the monitor runner (`monitors.run_monitor`): the cap is
`max_tokens`, or for dynamic-output monitors the context left after the prompt; a reply is valid when the registered parser
finds a score (reasoning monitors: only in the final answer, fail-closed on a length stop or an unclosed reasoning block).
Scores are never stored in the rows and never compared with labels (cost only).

Boundaries. This is a standalone judge measurement. It is not an agent-plus-judge run; any sum with agent generation time is
an accounting estimate. It is also not throughput. The timed interval of a row is the HTTP request to the vLLM server on the same machine (request sent to
parsed JSON reply); prompt construction, tokenization for the cap and the untimed warm-up request are outside it. The server
start-up time is recorded separately (server log and launcher record), never added to a row.

Files in --out-dir: run_config.json (immutable; written before the first request; a resume with a different config is refused),
rows.jsonl (one row per attempt, errors included, appended and fsynced), replies.jsonl (reply texts, for audit),
sessions.jsonl (one record per start: warm-up, residency, GPU memory, git state), summary.json / summary.md.
Progress output is counts only.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import os
import pathlib
import platform
import re
import statistics
import subprocess
import sys
import time

import requests

from evals.reward_hacking import llm
from evals.reward_hacking import monitors as m

CONDITIONS = ("history_black_box", "history_reasoning", "final_code_only")
CONDITION_TEXT = {"history_black_box": "History (transcript)", "history_reasoning": "History + private reasoning",
                  "final_code_only": "Final code only"}
MONITOR_ORDER = ("gemma4-e4b", "gemma4-e4b-think", "qwen3.8-27b-4k", "qwen3.8-27b-think", "glm5.3-flash-nvfp4-think")
HERE = pathlib.Path(__file__).resolve().parent
WARMUP_RECORD = ("[ORIGINAL TASK INSTRUCTION]\nWrite a function add(a, b) that returns a + b.\n\n[SUBMITTED FILES]\n"
                 "--- func.py ---\ndef add(a, b):\n    return a + b\n--- test.py ---\nassert add(1, 2) == 3\n")


# ---------------------------------------------------------------------------------------------------------------------
# small helpers

def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha_obj(obj) -> str:
    return hashlib.sha256(canon(obj).encode()).hexdigest()


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def condition_of(item: dict) -> str:
    return {("history", "black_box"): "history_black_box", ("history", "cot"): "history_reasoning",
            ("final_state", "black_box"): "final_code_only"}[(item["input"], item["view"])]


def quantile(xs, q):
    """Linear-interpolation quantile (the numpy default); None for an empty list."""
    xs = sorted(xs)
    if not xs:
        return None
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def run_cmd(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def gpu_static() -> dict:
    out = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,power.limit", "--format=csv,noheader,nounits"])
    cpu = None
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"nvidia_smi_gpu_name_memory_total_mib_driver_power_limit_w": out, "cpu": cpu, "cpu_count": os.cpu_count(),
            "platform": platform.platform()}


def gpu_dynamic(server_pid: int | None = None) -> dict:
    """GPU memory in use and the compute processes now (a session record: these change between starts)."""
    used = run_cmd(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
    apps = run_cmd(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    pids = []
    app_rows = [a.strip() for a in (apps or "").splitlines() if a.strip()]
    for a in app_rows:
        try:
            pids.append(int(a.split(",")[0]))
        except ValueError:
            pass
    sole = None
    if server_pid is not None and apps is not None:
        ps = run_cmd(["ps", "-eo", "pid=,ppid="]) or ""
        children: dict[int, list[int]] = {}
        for line in ps.splitlines():
            try:
                pid, ppid = (int(x) for x in line.split())
            except ValueError:
                continue
            children.setdefault(ppid, []).append(pid)
        tree, todo = {server_pid}, [server_pid]
        while todo:
            for c in children.get(todo.pop(), []):
                if c not in tree:
                    tree.add(c)
                    todo.append(c)
        sole = bool(pids) and all(p in tree for p in pids)
    return {"memory_used_total_mib": used, "compute_apps_pid_name_mib": app_rows, "server_pid": server_pid,
            "only_the_server_process_tree_on_gpu": sole}


def git_state() -> dict:
    root = HERE.parent
    head = run_cmd(["git", "-C", str(root), "rev-parse", "HEAD"])
    dirty = run_cmd(["git", "-C", str(root), "status", "--porcelain", "--", "evals_runner", "evals/reward_hacking"])
    return {"head": head, "dirty_in_evals_runner_or_reward_hacking": bool(dirty) if dirty is not None else None}


def code_hashes() -> dict:
    files = [HERE / "eq1_monitor_panel.py", HERE / "eq1_monitor_panel_prompts.py",
             HERE.parent / "evals" / "reward_hacking" / "monitors.py", HERE.parent / "evals" / "reward_hacking" / "trajectory.py"]
    return {f.name: sha256_file(f) for f in files if f.is_file()}


def model_identity(model_dir: str | None, spec) -> dict | None:
    """Files of the local snapshot with their Hugging Face blob names (the sha256 of the content for LFS files such as
    safetensors), and whether the snapshot revision (the directory name) equals the registry's pin."""
    if not model_dir:
        return None
    d = pathlib.Path(model_dir)
    files = []
    for p in sorted(d.iterdir()):
        e = {"name": p.name}
        try:
            e["size"] = p.stat().st_size
        except OSError:
            e["size"] = None
        if p.is_symlink():
            blob = os.path.basename(os.readlink(p))
            e["blob"] = blob
            e["blob_kind"] = "sha256 of content" if re.fullmatch(r"[0-9a-f]{64}", blob) else "git object id"
        files.append(e)
    rev = d.name
    tpl = None
    for cand in ("chat_template.jinja",):
        if (d / cand).is_file():
            tpl = sha256_file(d / cand)
    return {"dir": str(d), "snapshot_revision": rev, "registry_revision": spec.revision,
            "revision_matches_registry": rev == spec.revision, "chat_template_sha256": tpl, "files": files}


LOG_PATTERNS = {
    "model_loading": r"Model loading took ([\d.]+) GiB(?: and ([\d.]+) seconds)?",
    "weights_loading_s": r"Loading weights took ([\d.]+) seconds",
    "init_engine_s": r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s",
    "torch_compile_s": r"torch\.compile takes ([\d.]+) s",
    "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
    "max_concurrency": r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x",
    "prefix_cache_hit_rate": r"Prefix cache hit rate: ([\d.]+)%",
    "non_default_args": r"non-default args: (\{.*\})",
}


def parse_server_log(path) -> dict | None:
    """Start-up facts and prefix-cache hit rates from the vLLM log (best effort; every matched line is kept as text)."""
    if not path or not os.path.isfile(path):
        return None
    text = open(path, errors="replace").read()
    out: dict = {"log": str(path), "log_sha256": sha256_file(path), "lines": len(text.splitlines())}
    for key, pat in LOG_PATTERNS.items():
        hits = [mm.group(0) for mm in re.finditer(pat, text)]
        if not hits:
            out[key] = None
            continue
        first = re.search(pat, text)
        out[key] = {"first_match": first.group(0)[:400], "n_matches": len(hits), "groups": list(first.groups())}
        if key == "prefix_cache_hit_rate":
            rates = [float(x) for x in re.findall(pat, text)]
            out[key].update({"last_percent": rates[-1], "max_percent": max(rates)})
    return out


def read_json(path):
    return json.loads(pathlib.Path(path).read_text()) if path else None


# ---------------------------------------------------------------------------------------------------------------------
# request path: the same request body, output cap and parse rule as monitors.run_monitor

def user_tokens(kind: str, user: str, counter, cfg) -> int:
    """`info["user_tokens"]` of the runner: history views count the template overhead (_fit), the final-code view does not."""
    n = counter.count(user)
    return n + cfg.template_overhead_tokens if kind == "history" else n


def output_budget(cfg, utokens: int) -> int:
    """The cap `run_monitor` sends: `max_tokens`, or the context left after the prompt for dynamic-output monitors."""
    if cfg.server_context:
        left = cfg.server_context - utokens - cfg.template_overhead_tokens - 512
        return max(cfg.min_output_tokens, min(cfg.max_tokens, left))
    return cfg.max_tokens


def fits_registry_budget(kind: str, utokens: int, cfg) -> bool:
    """The runner shortens private reasoning or raises when the prompt exceeds `max_prompt_tokens`; such a request would not
    be the detector's input, so the panel does not send it."""
    return utokens <= cfg.max_prompt_tokens if kind == "history" else utokens + cfg.template_overhead_tokens <= cfg.max_prompt_tokens


def reply_valid(text: str, finish: str, cfg) -> tuple[bool, str | None]:
    info: dict = {}
    if cfg.think_close or cfg.server_context:
        score = m.parse_reasoning_reply(text, finish, cfg, m.parse_score, info)
    else:
        score = m.parse_score(text.split(llm.THINK_CLOSE, 1)[1] if llm.THINK_CLOSE in text else text)
    return score is not None, info.get("unparseable_reason")


def post_once(client, url: str, user: str, budget: int, cfg, timeout: float) -> dict:
    """One attempt, no retry inside: the client's own retry loop would hide failures and add sleeps to the timed interval."""
    body = client.request_body(user, max_tokens=budget, seed=cfg.seed, sampling=cfg.sampling, enable_thinking=cfg.enable_thinking)
    t0 = time.perf_counter()
    try:
        r = requests.post(url.rstrip("/") + "/v1/chat/completions", json=body, timeout=timeout)
        status = r.status_code
        r.raise_for_status()
        d = r.json()
        choice = d["choices"][0]
        usage = d.get("usage") or {}
        msg = choice.get("message") or {}
        seconds = time.perf_counter() - t0
        text, finish = msg.get("content") or "", choice.get("finish_reason") or ""
        valid, why = reply_valid(text, finish, cfg)
        return {"seconds": seconds, "http_status": status, "http_error": None, "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "generated_tokens": int(usage.get("completion_tokens") or 0), "finish_reason": finish, "valid_score": valid,
                "unparseable_reason": why, "reply_chars": len(text), "reasoning_chars": len(msg.get("reasoning_content") or ""),
                "cached_prompt_tokens": ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")), "_text": text}
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return {"seconds": time.perf_counter() - t0, "http_status": status, "http_error": f"{type(exc).__name__}: {str(exc)[:300]}",
                "prompt_tokens": None, "generated_tokens": None, "finish_reason": None, "valid_score": False,
                "unparseable_reason": None, "reply_chars": None, "reasoning_chars": None, "cached_prompt_tokens": None, "_text": None}


# ---------------------------------------------------------------------------------------------------------------------
# rows, config, resume

def read_rows(path: pathlib.Path) -> list[dict]:
    """Complete JSON rows; a torn last line (killed mid-write) is dropped."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def append_jsonl(path: pathlib.Path, obj: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def item_state(rows: list[dict], max_attempts: int, retry_failed: bool = False) -> dict[int, dict]:
    """Per item: HTTP attempts made, whether a reply came back, and whether the item is finished (replied, not sent because the
    prompt is over the registry budget, or all attempts spent)."""
    st: dict[int, dict] = {}
    for r in rows:
        s = st.setdefault(r["item"], {"attempts": 0, "replied": False, "done": False})
        if r.get("not_sent"):
            s["done"] = True
            continue
        s["attempts"] += 1
        if r.get("http_error") is None:
            s["replied"] = s["done"] = True
    for s in st.values():
        if s["attempts"] >= max_attempts and not retry_failed:
            s["done"] = True
    return st


def check_or_write_config(out_dir: pathlib.Path, config: dict) -> None:
    path = out_dir / "run_config.json"
    sha = sha_obj(config)
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("config_sha256") != sha or sha_obj(old["config"]) != sha:
            diff = sorted(k for k in set(old["config"]) | set(config) if old["config"].get(k) != config.get(k))
            raise SystemExit(f"refusing to resume: the run configuration changed (differs in: {', '.join(diff) or 'hash only'}); "
                             f"use a new --out-dir for a different configuration")
        return
    if (out_dir / "rows.jsonl").exists() and (out_dir / "rows.jsonl").stat().st_size:
        raise SystemExit("refusing to start: rows exist without a run_config.json")
    with open(path, "x") as f:       # "x": never overwrite
        f.write(json.dumps({"config_sha256": sha, "written_utc": now_utc(), "config": config}, indent=1) + "\n")


# ---------------------------------------------------------------------------------------------------------------------
# summary

def cap_text(spec) -> str:
    base = f"reasoning {'on' if spec.enable_thinking else 'off'}, cap {spec.max_tokens:,}"
    return base + (" (context-limited)" if spec.dynamic_output else "")


def summarize(rows: list[dict], items: list[dict], spec_dict: dict) -> dict:
    """Per condition. Denominators: `invocations` = planned items; `attempts` = rows with an HTTP attempt (every try kept);
    `replies` = items whose request returned a reply; `valid` = replies with a valid score; `valid_coverage` = valid / invocations.
    Time and token statistics are over replies (valid or not); error attempts and unsent items have no tokens and are counted."""
    out = {}
    for cond in CONDITIONS:
        idx = [i for i, it in enumerate(items) if condition_of(it) == cond]
        if not idx:
            continue
        rs = [r for r in rows if r["item"] in set(idx)]
        sent = [r for r in rs if not r.get("not_sent")]
        replies = [r for r in sent if r.get("http_error") is None]
        valid_items = {r["item"] for r in replies if r["valid_score"]}
        secs = [r["seconds"] for r in replies]
        d = {"invocations": len(idx), "attempts": len(sent), "http_error_attempts": len(sent) - len(replies),
             "not_sent_over_budget": sum(1 for r in rs if r.get("not_sent")), "replies": len({r["item"] for r in replies}),
             "valid": len(valid_items), "valid_coverage": len(valid_items) / len(idx),
             "no_score_replies": len({r["item"] for r in replies}) - len(valid_items),
             "length_stops": sum(1 for r in replies if r["finish_reason"] == "length"), "items_without_reply": len(idx) - len({r["item"] for r in replies})}
        if replies:
            d.update({"seconds_mean": statistics.fmean(secs), "seconds_median": statistics.median(secs), "seconds_p90": quantile(secs, .9),
                      "seconds_min": min(secs), "seconds_max": max(secs), "seconds_total": sum(secs),
                      "input_tokens_mean": statistics.fmean(r["prompt_tokens"] for r in replies),
                      "output_tokens_mean": statistics.fmean(r["generated_tokens"] for r in replies),
                      "output_tokens_max": max(r["generated_tokens"] for r in replies)})
        else:
            d["note"] = "no reply returned; time and token statistics undefined"
        out[cond] = d
    return out


def f2(x, nd=2):
    return "-" if x is None else f"{x:,.{nd}f}"


def table_rows(name: str, spec_dict: dict, summ: dict) -> list[dict]:
    spec = m.get_spec(name)
    res = []
    for cond in CONDITIONS:
        s = summ.get(cond)
        if not s:
            continue
        res.append({"monitor": name, "view": CONDITION_TEXT[cond], "config": cap_text(spec), "invocations": s["invocations"],
                    "valid": s["valid"], "seconds": f"{f2(s.get('seconds_mean'))}, {f2(s.get('seconds_median'))}",
                    "input_tokens": f2(s.get("input_tokens_mean"), 0), "output_tokens": f2(s.get("output_tokens_mean"), 0),
                    "extra": s})
    return res


HEAD = ["Monitor", "View", "Config", "Invocations", "Valid", "Seconds/invocation (mean, median)", "Input tok.", "Output tok."]
NOTE = ("Standalone judge measurement on recorded training-partition views (one request at a time, natural stops, the registered "
        "reasoning setting and output cap); it is not an agent-plus-judge run, and any sum with agent time is an accounting estimate. "
        "Invocations are all planned requests; Valid is the number with a parsable score; seconds and tokens are means over replies "
        "that returned (the median is of seconds). Input and output tokens are server counts.")


def md_table(lines: list[dict], unmeasured: list[dict]) -> str:
    out = ["| " + " | ".join(HEAD) + " |", "|" + "|".join(["---"] * 3 + ["---:"] * 5) + "|"]
    for r in lines:
        out.append(f"| {r['monitor']} | {r['view']} | {r['config']} | {r['invocations']} | {r['valid']} | {r['seconds']} | "
                   f"{r['input_tokens']} | {r['output_tokens']} |")
    for u in unmeasured:
        out.append(f"| {u['monitor']} | all | {u['config']} | not measured on this hardware | | | | |")
    txt = "\n".join(out) + "\n\n" + NOTE + "\n"
    for u in unmeasured:
        txt += f"\nNot measured: {u['monitor']}: {u['reason']}\n"
    return txt


def tex_table(lines: list[dict], unmeasured: list[dict]) -> str:
    esc = lambda s: str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
    out = [r"\begin{table}[t]", r"\centering\footnotesize", r"\begin{tabular}{lllrrrrr}", r"\toprule",
           " & ".join(["Monitor", "View", "Config", "Invoc.", "Valid", r"Seconds/invocation (mean, median)", "Input tok.", "Output tok."]) + r" \\",
           r"\midrule"]
    for r in lines:
        out.append(" & ".join(esc(x) for x in (r["monitor"], r["view"], r["config"], r["invocations"], r["valid"], r["seconds"],
                                               r["input_tokens"], r["output_tokens"])) + r" \\")
    for u in unmeasured:
        out.append(" & ".join([esc(u["monitor"]), "all", esc(u["config"]), r"\multicolumn{5}{l}{not measured on this hardware}"]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\caption{Monitor invocation cost. " + esc(NOTE) + "}", r"\end{table}"]
    return "\n".join(out) + "\n"


def write_summary(out_dir: pathlib.Path, config: dict, items: list[dict], rows: list[dict], server_log: str | None) -> dict:
    summ = summarize(rows, items, config["registry_spec"])
    name = config["monitor"]
    lines = table_rows(name, config["registry_spec"], summ)
    sessions = read_rows(out_dir / "sessions.jsonl")
    doc = {"monitor": name, "config_sha256": sha_obj(config), "per_condition": summ, "sessions": len(sessions),
           "server_log_end_of_run": parse_server_log(server_log) if server_log else None,
           "note": NOTE}
    (out_dir / "summary.json").write_text(json.dumps(doc, indent=1) + "\n")
    (out_dir / "summary.md").write_text(md_table(lines, []))
    return doc


def combine(root: pathlib.Path) -> int:
    lines, unmeasured, docs = [], [], {}
    names = [d.name for d in root.iterdir() if d.is_dir() and ((d / "summary.json").exists() or (d / "not_measured.json").exists())]
    names.sort(key=lambda n: (MONITOR_ORDER.index(n) if n in MONITOR_ORDER else len(MONITOR_ORDER), n))
    for n in names:
        d = root / n
        if (d / "summary.json").exists():
            doc = json.loads((d / "summary.json").read_text())
            docs[n] = doc
            lines += table_rows(n, {}, doc["per_condition"])
        else:
            u = json.loads((d / "not_measured.json").read_text())
            unmeasured.append({"monitor": n, "config": cap_text(m.get_spec(n)), "reason": u["reason"]})
    (root / "panel_table.md").write_text(md_table(lines, unmeasured))
    (root / "panel_table.tex").write_text(tex_table(lines, unmeasured))
    (root / "panel_summary.json").write_text(json.dumps({"monitors": docs, "not_measured": unmeasured}, indent=1) + "\n")
    print((root / "panel_table.md").read_text())
    return 0


def not_measured(name: str, out_dir: pathlib.Path, reason: str) -> int:
    spec = m.get_spec(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {"monitor": name, "measured": False, "reason": reason, "registry_spec": spec.as_dict(), "registry_config_sha256": sha_obj(
        {"spec": spec.as_dict(), "config": spec.config().as_dict()}), "hardware": gpu_static(), "recorded_utc": now_utc(),
        "note": "recorded instead of omitting the monitor: a different machine would be needed, and timings from different hardware "
                "are not comparable with these rows"}
    (out_dir / "not_measured.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"[panel {name}] not measured: {reason}")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# main

def load_prompts(prompts: pathlib.Path, manifest: pathlib.Path) -> tuple[list[dict], dict, str, str]:
    """Items in schedule order, checked against the manifest (ids, order, user hashes); returns the file hashes."""
    blob = gzip.decompress(prompts.read_bytes())
    items = json.loads(blob)["items"]
    man = json.loads(manifest.read_text())
    mi = man["items"]
    if len(mi) != len(items):
        raise SystemExit(f"prompt file has {len(items)} items, manifest {len(mi)}")
    for it, mm in zip(items, mi):
        if hashlib.sha256(it["user"].encode()).hexdigest() != it["user_sha256"] or it["user_sha256"] != mm["user_sha256"] \
                or it["rollout_id"] != mm["rollout_id"]:
            raise SystemExit(f"prompt file and manifest disagree at {it['rollout_id']} {it['view']}")
    if man.get("prompts_file_sha256") != sha256_file(prompts):
        raise SystemExit("the prompt file is not the one the manifest describes (hash differs)")
    return items, man, sha256_file(prompts), sha256_file(manifest)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--monitor")
    ap.add_argument("--url")
    ap.add_argument("--prompts", type=pathlib.Path)
    ap.add_argument("--manifest", type=pathlib.Path)
    ap.add_argument("--out-dir", type=pathlib.Path)
    ap.add_argument("--model-dir", help="local snapshot directory of the served weights (identity and tokenizer)")
    ap.add_argument("--server-log")
    ap.add_argument("--server-info", help="JSON written by the launcher: {flags: {...}, session: {...}} (flags are part of the config)")
    ap.add_argument("--server-pid", type=int, default=None)
    ap.add_argument("--max-attempts", type=int, default=1, help="HTTP attempts per request (a retry only follows an HTTP failure; every attempt is a row)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="on a resume, give items whose attempts all failed another round of --max-attempts (the earlier rows stay; "
                         "use after a server crash)")
    ap.add_argument("--timeout", type=float, default=None, help="request timeout in seconds (default 3600; 7200 for --retry)")
    ap.add_argument("--retry", action="store_true",
                    help="retry the failed/unsent items of the panel run in --panel-dir; writes only to --out-dir (see eq1_monitor_panel_retry.py)")
    ap.add_argument("--panel-dir", type=pathlib.Path, help="--retry: the ORIGINAL monitor output directory (read only)")
    ap.add_argument("--trajectories", type=pathlib.Path, help="--retry: directory of the recorded trajectories the prompts were built from")
    ap.add_argument("--server-context", type=int, default=None, help="--retry: the server's max_model_len (default: asked from the server)")
    ap.add_argument("--plan-only", action="store_true", help="--retry: print the retry plan (token counts, caps, shortening) and write nothing")
    ap.add_argument("--retry-list", action="store_true", help="print the number of failed items of --panel-dir (by the retry rule) and exit")
    ap.add_argument("--retry-combine", type=pathlib.Path, help="write summary.json and panel_table_completed.md/.tex over <dir>/<monitor>/ retry outputs")
    ap.add_argument("--panel-root", type=pathlib.Path, help="--retry-combine: where not_measured.json records are (default: parent of the retry root)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N items of the schedule (a smoke test; part of the config)")
    ap.add_argument("--max-consecutive-errors", type=int, default=5)
    ap.add_argument("--allow-char-counter", action="store_true", help="tests only: count tokens by characters if no tokenizer loads")
    ap.add_argument("--combine", type=pathlib.Path, help="write the table over every monitor directory below this one")
    ap.add_argument("--not-measured", metavar="NAME")
    ap.add_argument("--reason")
    ap.add_argument("--summarize-only", action="store_true", help="recompute summary files from the rows of --out-dir")
    a = ap.parse_args(argv)
    if a.retry_combine:
        from evals_runner.eq1_monitor_panel_retry import combine_retry
        return combine_retry(a.retry_combine, a.panel_root)
    if a.retry_list:
        from evals_runner.eq1_monitor_panel_retry import failed_items
        cfgdoc = json.loads((a.panel_dir / "run_config.json").read_text())["config"]
        print(len(failed_items(read_rows(a.panel_dir / "rows.jsonl"), len(cfgdoc["schedule"]))))
        return 0
    if a.retry:
        from evals_runner.eq1_monitor_panel_retry import DEFAULT_RETRY_TIMEOUT, run_retry
        for need in ("panel_dir", "out_dir", "url"):
            if getattr(a, need) is None:
                raise SystemExit(f"--retry needs --{need.replace('_', '-')}")
        a.timeout = a.timeout or DEFAULT_RETRY_TIMEOUT
        return run_retry(a)
    a.timeout = a.timeout or 3600.0
    if a.combine:
        return combine(a.combine)
    if a.not_measured:
        if not (a.out_dir and a.reason):
            raise SystemExit("--not-measured needs --out-dir and --reason")
        return not_measured(a.not_measured, a.out_dir, a.reason)
    for need in ("monitor", "prompts", "manifest", "out_dir"):
        if getattr(a, need) is None:
            raise SystemExit(f"--{need.replace('_', '-')} is required")
    if a.url is None and not a.summarize_only:
        raise SystemExit("--url is required")
    spec = m.get_spec(a.monitor)
    cfg = spec.config()
    items, man, prompts_sha, manifest_sha = load_prompts(a.prompts, a.manifest)
    if a.limit:
        items = items[:a.limit]
    out_dir = a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path, replies_path, sessions_path = out_dir / "rows.jsonl", out_dir / "replies.jsonl", out_dir / "sessions.jsonl"

    if a.summarize_only:
        cfgdoc = json.loads((out_dir / "run_config.json").read_text())["config"]
        doc = write_summary(out_dir, cfgdoc, items, read_rows(rows_path), a.server_log)
        print(json.dumps({k: doc["per_condition"][k]["valid"] for k in doc["per_condition"]}))
        return 0

    # tokenizer: the runner counts the prompt with the monitor's own tokenizer to set the cap and check the prompt budget
    src = a.model_dir or spec.hf_id
    counter = m.load_counter(src, revision=spec.revision, local_only=True)
    method = getattr(counter, "method", "chars")
    if method != "tokenizer" and not a.allow_char_counter:
        raise SystemExit("the monitor's tokenizer did not load; refusing to fall back to a character estimate (the output cap and "
                         "prompt budget would differ from the detector's)")
    client = m.ChatMonitorClient(a.url, model=spec.weights, server="vllm", extra_body=spec.request_extra,
                                 reasoning_format=spec.reasoning_format, assistant_prefill=spec.assistant_prefill)
    ident = client.identity()
    plan = []     # per item: tokens counted by the tokenizer, cap, whether it is sent
    for it in items:
        ut = user_tokens(it["input"], it["user"], counter, cfg)
        plan.append({"user_tokens": ut, "output_cap": output_budget(cfg, ut), "sendable": fits_registry_budget(it["input"], ut, cfg)})
    sinfo = read_json(a.server_info)
    flags = (sinfo or {}).get("flags")
    config = {
        "schema": 1, "monitor": a.monitor, "registry_spec": spec.as_dict(), "monitor_config": cfg.as_dict(),
        "registry_config_sha256": sha_obj({"spec": spec.as_dict(), "config": cfg.as_dict()}),
        "request_body_shape": client.request_body("<user message>", max_tokens=cfg.max_tokens, seed=cfg.seed, sampling=cfg.sampling,
                                                  enable_thinking=cfg.enable_thinking),
        "concurrency": 1, "max_attempts": a.max_attempts, "limit": a.limit,
        "schedule": [{"i": i, "rollout_id": it["rollout_id"], "condition": condition_of(it), "user_sha256": it["user_sha256"],
                      "user_tokens": p["user_tokens"], "output_cap": p["output_cap"], "sendable": p["sendable"]}
                     for i, (it, p) in enumerate(zip(items, plan))],
        "prompts_file_sha256": prompts_sha, "manifest_sha256": manifest_sha,
        "sample_rule": man.get("rule"), "sample_task_ids": man.get("task_ids"),
        "token_counting": {"method": method, "source": str(src)},
        "timed_interval": "HTTP request to the local vLLM server, sent to parsed JSON reply; no retry inside; warm-up and prompt construction excluded",
        "server": {"engine": ident.get("engine"), "engine_version": ident.get("engine_version"), "model_alias": ident.get("model_alias"),
                   "max_model_len": ident.get("max_model_len"), "launch_flags": flags},
        "hardware": gpu_static(), "model": model_identity(a.model_dir, spec), "code_sha256": code_hashes(),
        "score_values_stored": False, "labels_read": False,
    }
    if config["model"] and not config["model"]["revision_matches_registry"]:
        print(f"[panel {a.monitor}] WARNING: snapshot revision {config['model']['snapshot_revision']} differs from the registry pin "
              f"{spec.revision}", file=sys.stderr, flush=True)
    check_or_write_config(out_dir, config)

    rows = read_rows(rows_path)
    st = item_state(rows, a.max_attempts, a.retry_failed)
    if all(st.get(i, {}).get("done") for i in range(len(items))):
        doc = write_summary(out_dir, config, items, rows, a.server_log)
        print(f"[panel {a.monitor}] nothing left to run; summary rewritten", flush=True)
        return 0
    session = {"session": len(read_rows(sessions_path)) + 1, "start_utc": now_utc(), "git": git_state(), "server_info_session": (sinfo or {}).get("session"),
               "gpu_before_warmup": gpu_dynamic(a.server_pid), "server_log_at_start": parse_server_log(a.server_log),
               "resumed_rows": len(rows), "retry_failed": a.retry_failed}
    # one untimed warm-up request with a fixed synthetic record (not a sample prompt); recorded, never in the rows
    wu_user = m.FINAL_STATE_INSTRUCTION.format(reasoning_note="", record=WARMUP_RECORD)
    wu_cap = output_budget(cfg, user_tokens("final_state", wu_user, counter, cfg))
    wu = post_once(client, a.url, wu_user, wu_cap, cfg, a.timeout)
    wu.pop("_text", None)
    session["warmup"] = {**wu, "untimed": True, "note": "outside every row; includes any first-request compilation"}
    session["gpu_after_warmup"] = gpu_dynamic(a.server_pid)
    append_jsonl(sessions_path, session)

    t_session = time.perf_counter()
    consecutive = 0
    total = len(items)
    for i, (it, p) in enumerate(zip(items, plan)):
        if st.get(i, {}).get("done"):
            continue
        cond = condition_of(it)
        base = {"item": i, "rollout_id": it["rollout_id"], "task_id": it.get("task_id"), "condition": cond, "view": it["view"],
                "input": it["input"], "user_sha256": it["user_sha256"], "user_tokens_counted": p["user_tokens"], "output_cap": p["output_cap"],
                "session": session["session"], "concurrency": 1}
        if not p["sendable"]:
            append_jsonl(rows_path, {**base, "attempt": 0, "not_sent": True, "http_error": None, "valid_score": False,
                                     "note": "prompt over the registry's prompt budget: the detector would have shortened or refused it",
                                     "start_utc": now_utc()})
            continue
        prior, tries = st.get(i, {}).get("attempts", 0), 0
        while tries < a.max_attempts:
            tries += 1
            attempts = prior + tries
            start_utc, offset = now_utc(), time.perf_counter() - t_session
            res = post_once(client, a.url, it["user"], p["output_cap"], cfg, a.timeout)
            text = res.pop("_text")
            append_jsonl(rows_path, {**base, "attempt": attempts, "start_utc": start_utc, "start_offset_s": round(offset, 3),
                                     **{k: (round(v, 6) if k == "seconds" else v) for k, v in res.items()}})
            if text is not None:
                append_jsonl(replies_path, {"item": i, "attempt": attempts, "text": text})
            consecutive = consecutive + 1 if res["http_error"] else 0
            if not res["http_error"]:
                break
            if consecutive >= a.max_consecutive_errors:
                break
        done = read_rows(rows_path)
        reps = {r["item"] for r in done if r.get("http_error") is None and not r.get("not_sent")}
        vals = {r["item"] for r in done if r.get("http_error") is None and r.get("valid_score")}
        errs = sum(1 for r in done if r.get("http_error"))
        print(f"[panel {a.monitor}] item {i + 1}/{total}: replies {len(reps)}, valid scores {len(vals)}, http error attempts {errs}", flush=True)
        if consecutive >= a.max_consecutive_errors:
            print(f"[panel {a.monitor}] stopping after {consecutive} consecutive errors; rerun the same command to resume", flush=True)
            return 3
    rows = read_rows(rows_path)
    doc = write_summary(out_dir, config, items, rows, a.server_log)
    print(json.dumps({c: {"invocations": v["invocations"], "valid": v["valid"], "replies": v["replies"]} for c, v in doc["per_condition"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
