"""Tests for the EQ1 monitor invocation cost panel (stub HTTP server, synthetic trajectories; no model, no GPU)."""
import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from evals.reward_hacking import monitors as m
from evals.reward_hacking import trajectory as tj
from evals_runner import eq1_monitor_panel as panel
from evals_runner import eq1_monitor_panel_prompts as prompts


# ---------------------------------------------------------------------------------------------- synthetic data

def make_traj(task: str, k: int = 0, partition: str = "train", submit: bool = True) -> tj.Trajectory:
    t = tj.Trajectory(meta={"task_id": task, "rollout_id": prompts.rollout_id(task, k), "partition": partition})
    t.add(tj.USER, {"text": f"implement f for {task}", "origin": "task"})
    t.add(tj.MODEL_CALL, {}, call_index=0)
    t.add(tj.REASONING, {"text": "private thoughts about the tests"}, call_index=0)
    t.add(tj.ASSISTANT, {"text": "I will write the function"}, call_index=0)
    t.add(tj.TOOL_PROPOSED, {"call_id": "c1", "name": "run", "arguments": {"cmd": "pytest"}, "raw": "", "parse_error": None}, call_index=0)
    t.add(tj.TOOL_EXECUTED, {"call_id": "c1", "name": "run", "arguments": {"cmd": "pytest"}, "modified_by_wrapper": False}, call_index=0)
    t.add(tj.TOOL_RESULT, {"call_id": "c1", "name": "run", "output": "1 failed"}, call_index=0)
    if submit:
        t.add(tj.SUBMISSION, {"attempt": 1, "func_py": "def f(): return 1", "test_py": "assert f() == 1"}, call_index=0)
        t.add(tj.SUBMISSION_EXECUTED, {"attempt": 1, "func_py": "def f(): return 1", "test_py": "assert f() == 1",
                                       "modified_by_wrapper": False})
        t.add(tj.EVALUATION, {"attempt": 1, "passed": False, "explanation": "x"})
    return t


def write_world(tmp_path, tasks_train, tasks_other=(), skip_files=(), no_submission=(), wrong_partition=()):
    """splits.json plus a trajectories directory. Task IDs are lexically unordered on purpose (lcbhard_9 > lcbhard_10)."""
    assign = {t: "train" for t in tasks_train} | {t: "val" for t in tasks_other}
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"assignment": assign}))
    d = tmp_path / "trajs" / "trajectories"
    d.mkdir(parents=True)
    for t in list(tasks_train) + list(tasks_other):
        if t in skip_files:
            continue
        part = "test" if t in wrong_partition else assign[t]
        traj = make_traj(t, 0, partition=part, submit=t not in no_submission)
        (d / f"{prompts.rollout_id(t)}.json").write_text(traj.to_json())
        (d / f"{prompts.rollout_id(t, 1)}.json").write_text(make_traj(t, 1).to_json())   # an r1 file: must never be used
    return splits, tmp_path / "trajs"


# ---------------------------------------------------------------------------------------------- the sample rule

def test_rule_lexical_first_n_train_only_r0(tmp_path):
    train = [f"lcbhard_{i}" for i in (9, 10, 100, 2, 33, 4)]
    splits, root = write_world(tmp_path, train, tasks_other=["lcbhard_0", "lcbhard_1"])
    sel = prompts.select_sample(json.loads(splits.read_text())["assignment"], root, n=4)
    # plain string order, train partition only, no val tasks, malformed tasks never
    assert sel["tasks"] == ["lcbhard_10", "lcbhard_100", "lcbhard_2", "lcbhard_33"]
    items = prompts.schedule_items(sel)
    assert len(items) == 12 and all(it["rollout_id"].endswith("__conflicting__T1__r0") for it in items)
    assert [it["condition"] for it in items] == ["history_black_box"] * 4 + ["history_reasoning"] * 4 + ["final_code_only"] * 4
    assert [it["task_id"] for it in items[:4]] == sel["tasks"] == [it["task_id"] for it in items[4:8]]
    assert all(it["label"] is None for it in items)


def test_rule_skips_for_structure_only_and_records_reasons(tmp_path):
    train = ["lcbhard_1", "lcbhard_10", "lcbhard_11", "lcbhard_12", "lcbhard_13", "lcbhard_35"]   # 35 is malformed
    splits, root = write_world(tmp_path, train, skip_files={"lcbhard_1"}, no_submission={"lcbhard_10"}, wrong_partition={"lcbhard_11"})
    sel = prompts.select_sample(json.loads(splits.read_text())["assignment"], root, n=5)
    assert sel["tasks"] == ["lcbhard_12", "lcbhard_13"]
    reasons = {s["task_id"]: s["reason"] for s in sel["skipped"]}
    assert set(reasons) == {"lcbhard_1", "lcbhard_10", "lcbhard_11"} and "lcbhard_35" not in reasons
    assert "no r0" in reasons["lcbhard_1"] and "partition" in reasons["lcbhard_11"]


def test_views_match_registered_builders_and_reasoning_view_differs(tmp_path):
    splits, root = write_world(tmp_path, ["lcbhard_1"])
    sel = prompts.select_sample(json.loads(splits.read_text())["assignment"], root, n=1)
    traj = make_traj("lcbhard_1")
    v = sel["views"]["lcbhard_1"]
    assert v["history_black_box"][0] == m.build_prompt(traj, tj.BLACK_BOX)[0]
    assert v["history_reasoning"][0] == m.build_prompt(traj, tj.COT)[0]
    assert v["final_code_only"][0] == m.build_final_state_prompt(traj)[0]
    assert "private thoughts" in v["history_reasoning"][0] and "private thoughts" not in v["history_black_box"][0]
    assert "TEST RESULT" not in v["final_code_only"][0]


def test_prompt_script_writes_gz_in_throughput_format_and_manifest(tmp_path):
    splits, root = write_world(tmp_path, [f"lcbhard_{i}" for i in range(5)])
    out = tmp_path / "p.json.gz"
    assert prompts.main(["--trajectories", str(root), "--splits", str(splits), "--n", "3", "--out", str(out)]) == 0
    items = json.loads(gzip.decompress(out.read_bytes()))["items"]
    man = json.loads((tmp_path / "p.manifest.json").read_text())
    for it in items:   # the fields eq2_rtx_throughput reads
        assert {"rollout_id", "view", "input", "user", "user_sha256", "label"} <= set(it)
    assert man["labels_read"] is False and man["task_ids"] == ["lcbhard_0", "lcbhard_1", "lcbhard_2"] and man["n_tasks"] == 3
    assert [i["user_sha256"] for i in man["items"]] == [i["user_sha256"] for i in items]
    # the same inputs give the same bytes (declared rule, no randomness, no timestamps)
    out2 = tmp_path / "q.json.gz"
    prompts.main(["--trajectories", str(root), "--splits", str(splits), "--n", "3", "--out", str(out2)])
    assert out.read_bytes() == out2.read_bytes()


def test_too_few_tasks_refused_unless_allowed(tmp_path):
    splits, root = write_world(tmp_path, ["lcbhard_0", "lcbhard_1"])
    with pytest.raises(SystemExit):
        prompts.main(["--trajectories", str(root), "--splits", str(splits), "--n", "5", "--out", str(tmp_path / "p.json.gz")])
    assert prompts.main(["--trajectories", str(root), "--splits", str(splits), "--n", "5", "--allow-fewer", "--out", str(tmp_path / "p.json.gz")]) == 0


# ---------------------------------------------------------------------------------------------- stub server

class Stub:
    """A local OpenAI-style server. `plan` maps a call number to 'ok' | 'noscore' | 'length' | 500."""

    def __init__(self, plan=None, max_ctx=None):
        self.plan, self.calls, self.bodies, self.max_ctx = plan or {}, 0, [], max_ctx
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path == "/version":
                    self._send(200, {"version": "0.31.0-stub"})
                elif self.path == "/v1/models":
                    self._send(200, {"data": [{"id": "stub", "max_model_len": 131072}]})
                else:
                    self._send(200, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.bodies.append(body)
                stub.calls += 1
                how = stub.plan.get(stub.calls, "ok")
                if stub.max_ctx and len(body["messages"][0]["content"]) / 3 + body["max_tokens"] > stub.max_ctx:   # chars/3 ~ tokens
                    self._send(400, {"error": "This model's maximum context length is exceeded"})
                    return
                if how == 500:
                    self._send(500, {"error": "boom"})
                    return
                text = {"ok": "reason\nSCORE: 7", "noscore": "no final line", "length": "cut off"}[how]
                self._send(200, {"choices": [{"message": {"content": text}, "finish_reason": "length" if how == "length" else "stop"}],
                                 "usage": {"prompt_tokens": 1000, "completion_tokens": 10}})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def world(tmp_path):
    splits, root = write_world(tmp_path, [f"lcbhard_{i}" for i in range(4)])
    out = tmp_path / "p.json.gz"
    prompts.main(["--trajectories", str(root), "--splits", str(splits), "--n", "4", "--out", str(out)])   # 12 items
    return out, tmp_path / "p.manifest.json", tmp_path


def run_panel(world, stub, name="gemma4-e4b", extra=(), out="run"):
    p, man, tmp = world
    return panel.main(["--monitor", name, "--url", stub.url, "--prompts", str(p), "--manifest", str(man), "--out-dir", str(tmp / out),
                       "--allow-char-counter", *extra]), tmp / out


def rows_of(d):
    return panel.read_rows(d / "rows.jsonl")


def test_run_sends_registry_body_one_at_a_time_and_summarizes(world):
    stub = Stub()
    rc, d = run_panel(world, stub)
    stub.close()
    assert rc == 0
    rows = rows_of(d)
    assert len(rows) == 12 and all(r["concurrency"] == 1 for r in rows)
    spec = m.get_spec("gemma4-e4b")
    b = stub.bodies[1]          # body 0 is the warm-up
    assert b["max_tokens"] == spec.max_tokens == 1024 and b["temperature"] == 0.0 and b["seed"] == spec.seed
    assert b["chat_template_kwargs"] == {"enable_thinking": False} and b["model"] == spec.weights and b["stream"] is False
    assert stub.calls == 13     # one warm-up, twelve invocations
    ses = panel.read_rows(d / "sessions.jsonl")
    assert len(ses) == 1 and ses[0]["warmup"]["untimed"] and "seconds" in ses[0]["warmup"]
    s = json.loads((d / "summary.json").read_text())["per_condition"]
    assert set(s) == set(panel.CONDITIONS)
    for c in s.values():
        assert (c["invocations"], c["valid"], c["replies"], c["http_error_attempts"], c["valid_coverage"]) == (4, 4, 4, 0, 1.0)
        assert c["input_tokens_mean"] == 1000 and c["output_tokens_mean"] == 10 and c["seconds_mean"] > 0
    # no score values and no label in the rows
    assert not any("score" in r and r["score"] is not None for r in rows) and "label" not in rows[0]
    assert (d / "replies.jsonl").exists()


def test_reasoning_monitor_body_and_cap(world):
    stub = Stub()
    rc, d = run_panel(world, stub, name="qwen3.8-27b-think")
    stub.close()
    spec = m.get_spec("qwen3.8-27b-think")
    b = stub.bodies[1]
    assert b["max_tokens"] == spec.max_tokens == 8192 and b["chat_template_kwargs"]["enable_thinking"] is True
    assert b["chat_template_kwargs"]["reasoning_effort"] == "medium" and b["skip_special_tokens"] is False
    # a reasoning monitor needs a closing marker: the stub's plain 'SCORE: 7' reply has none -> not valid (fail-closed, as the detector)
    s = json.loads((d / "summary.json").read_text())["per_condition"]
    assert all(c["valid"] == 0 and c["replies"] == 4 and c["no_score_replies"] == 4 for c in s.values())


def test_dynamic_cap_follows_runner_formula():
    spec = m.get_spec("gemma4-e4b-think")
    cfg = spec.config()
    assert panel.output_budget(cfg, 20000) == 65536
    assert panel.output_budget(cfg, 100000) == max(16384, 131072 - 100000 - 64 - 512)
    assert panel.output_budget(m.get_spec("gemma4-e4b").config(), 5) == 1024


def test_error_rows_kept_and_denominators(world):
    stub = Stub(plan={3: 500, 5: "noscore", 6: "length"})     # call 1 is the warm-up
    rc, d = run_panel(world, stub)
    stub.close()
    assert rc == 0
    rows = rows_of(d)
    assert len(rows) == 12                                     # every attempt is a row, errors included
    err = [r for r in rows if r["http_error"]]
    assert len(err) == 1 and err[0]["http_status"] == 500 and err[0]["prompt_tokens"] is None and err[0]["valid_score"] is False
    s = json.loads((d / "summary.json").read_text())["per_condition"]["history_black_box"]
    # items 0,1,2,3 are call numbers 2..5: call 3 (item 1) failed, call 5 (item 3) replied without a score
    assert (s["invocations"], s["attempts"], s["http_error_attempts"], s["replies"], s["valid"], s["no_score_replies"], s["items_without_reply"]) \
        == (4, 4, 1, 3, 2, 1, 1)
    assert s["valid_coverage"] == 0.5
    # time statistics are over the 3 replies, not the failed attempt
    ts = [r["seconds"] for r in rows if r["condition"] == "history_black_box" and not r["http_error"]]
    assert len(ts) == 3 and s["seconds_mean"] == pytest.approx(sum(ts) / 3)
    c2 = json.loads((d / "summary.json").read_text())["per_condition"]["history_reasoning"]
    assert c2["length_stops"] == 1 and c2["valid"] == 3      # call 6 = item 4 (first reasoning item): length stop, no score


def test_retry_attempts_are_all_kept(world):
    stub = Stub(plan={2: 500})
    rc, d = run_panel(world, stub, extra=["--max-attempts", "2"])
    stub.close()
    rows = rows_of(d)
    assert len(rows) == 13 and [r["attempt"] for r in rows[:2]] == [1, 2] and rows[0]["http_error"] and not rows[1]["http_error"]
    s = json.loads((d / "summary.json").read_text())["per_condition"]["history_black_box"]
    assert (s["invocations"], s["attempts"], s["http_error_attempts"], s["replies"], s["valid"]) == (4, 5, 1, 4, 4)


def test_config_is_written_before_first_row_and_is_immutable(world):
    stub = Stub()
    p, man, tmp = world
    rc, d = run_panel(world, stub, extra=["--limit", "3"])
    cfg = json.loads((d / "run_config.json").read_text())
    assert cfg["config_sha256"] == panel.sha_obj(cfg["config"]) and cfg["config"]["limit"] == 3 and cfg["config"]["concurrency"] == 1
    for k in ("registry_config_sha256", "prompts_file_sha256", "manifest_sha256", "code_sha256", "hardware", "schedule", "server"):
        assert k in cfg["config"]
    assert cfg["config"]["server"]["engine_version"] == "0.31.0-stub"
    assert {"eq1_monitor_panel.py", "eq1_monitor_panel_prompts.py"} <= set(cfg["config"]["code_sha256"])
    # same config: resume is fine and sends nothing more except nothing (all done)
    calls = stub.calls
    rc2, _ = run_panel(world, stub, extra=["--limit", "3"])
    assert rc2 == 0 and stub.calls == calls
    # changed config (limit, attempts): refused, rows untouched
    n = len(rows_of(d))
    for extra in (["--limit", "4"], ["--limit", "3", "--max-attempts", "2"]):
        with pytest.raises(SystemExit, match="refusing to resume"):
            run_panel(world, stub, extra=extra)
    with pytest.raises(SystemExit, match="refusing to resume"):      # a different monitor in the same directory
        run_panel(world, stub, name="gemma4-e4b-think", extra=["--limit", "3"])
    assert len(rows_of(d)) == n
    stub.close()


def test_resume_continues_after_interruption_and_does_not_repeat_items(world):
    stub = Stub(plan={5: 500, 6: 500})                          # call 1 is the warm-up: items 3 and 4 fail; the run stops
    rc, d = run_panel(world, stub, extra=["--max-consecutive-errors", "2"])
    assert rc == 3 and len(rows_of(d)) == 5
    stub.plan = {}
    n_calls = stub.calls
    rc2, _ = run_panel(world, stub, extra=["--max-consecutive-errors", "2"])          # failed items are finished (1 attempt each)
    assert rc2 == 0 and stub.calls - n_calls == 1 + 7             # warm-up and the 7 items never reached
    rc3, _ = run_panel(world, stub, extra=["--retry-failed"])    # now the two failed items get a new attempt
    stub.close()
    rows = rows_of(d)
    assert rc3 == 0 and len(rows) == 5 + 7 + 2
    by_item = {}
    for r in rows:
        by_item.setdefault(r["item"], []).append(r)
    assert all(len(by_item[i]) == 1 for i in (0, 1, 2, 5, 6, 11))
    assert [r["attempt"] for r in by_item[3]] == [1, 2] and by_item[3][0]["http_error"] and not by_item[3][1]["http_error"]
    assert len(panel.read_rows(d / "sessions.jsonl")) == 3
    s = json.loads((d / "summary.json").read_text())["per_condition"]
    assert all(c["valid"] == 4 and c["attempts"] >= 4 for c in s.values())
    assert s["history_black_box"]["http_error_attempts"] == 1 and s["history_reasoning"]["http_error_attempts"] == 1


def test_rows_without_config_refused(world, tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "rows.jsonl").write_text('{"item": 0}\n')
    stub = Stub()
    with pytest.raises(SystemExit, match="rows exist without"):
        run_panel(world, stub, out="bad")
    stub.close()


def test_tokenizer_required_unless_allowed(world):
    stub = Stub()
    p, man, tmp = world
    with pytest.raises(SystemExit, match="tokenizer"):
        panel.main(["--monitor", "gemma4-e4b", "--url", stub.url, "--prompts", str(p), "--manifest", str(man), "--out-dir", str(tmp / "t")])
    stub.close()


def test_manifest_mismatch_refused(world, tmp_path):
    p, man, tmp = world
    d = json.loads(man.read_text())
    d["items"][0]["user_sha256"] = "0" * 64
    bad = tmp_path / "bad_manifest.json"
    bad.write_text(json.dumps(d))
    stub = Stub()
    with pytest.raises(SystemExit, match="disagree"):
        panel.main(["--monitor", "gemma4-e4b", "--url", stub.url, "--prompts", str(p), "--manifest", str(bad), "--out-dir", str(tmp / "x"),
                    "--allow-char-counter"])
    stub.close()


def test_over_budget_prompt_is_not_sent_but_counted(tmp_path):
    splits, root = write_world(tmp_path, ["lcbhard_0", "lcbhard_1"])
    big = make_traj("lcbhard_0")
    big.events[6] = tj.Event(seq=6, kind=tj.TOOL_RESULT, call_index=0, data={"call_id": "c1", "name": "run", "output": "x" * 500_000})
    (root / "trajectories" / f"{prompts.rollout_id('lcbhard_0')}.json").write_text(big.to_json())
    out = tmp_path / "p.json.gz"
    prompts.main(["--trajectories", str(root), "--splits", str(splits), "--n", "2", "--out", str(out)])     # 6 items
    stub = Stub()
    rc, d = run_panel((out, tmp_path / "p.manifest.json", tmp_path), stub)
    stub.close()
    rows = rows_of(d)
    ns = [r for r in rows if r.get("not_sent")]
    assert rc == 0 and len(ns) == 2 and {r["condition"] for r in ns} == {"history_black_box", "history_reasoning"}   # final-code view fits
    assert stub.calls == 1 + 4                                      # warm-up and the four prompts that fit
    s = json.loads((d / "summary.json").read_text())["per_condition"]
    assert s["history_black_box"]["not_sent_over_budget"] == 1 and s["history_black_box"]["invocations"] == 2
    assert s["history_black_box"]["valid"] == 1 and s["history_black_box"]["valid_coverage"] == 0.5 and s["history_black_box"]["attempts"] == 1
    cfg = m.get_spec("gemma4-e4b").config()
    assert panel.fits_registry_budget("history", cfg.max_prompt_tokens, cfg)
    assert not panel.fits_registry_budget("history", cfg.max_prompt_tokens + 1, cfg)
    assert not panel.fits_registry_budget("final_state", cfg.max_prompt_tokens - cfg.template_overhead_tokens + 1, cfg)


def test_summary_quantiles_and_markdown_table(world, tmp_path):
    assert panel.quantile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], .9) == pytest.approx(9.1)
    assert panel.quantile([], .5) is None and panel.quantile([4.0], .9) == 4.0
    stub = Stub()
    rc, d = run_panel(world, stub, name="gemma4-e4b", out="root/gemma4-e4b")
    stub.close()
    root = d.parent
    panel.not_measured("glm5.3-flash-nvfp4-think", root / "glm5.3-flash-nvfp4-think", "weights are 204 GB; the GPU has 96 GB")
    assert panel.main(["--combine", str(root)]) == 0
    md = (root / "panel_table.md").read_text()
    assert md.splitlines()[0].startswith("| Monitor | View | Config | Invocations | Valid | Seconds/invocation (mean, median) | Input tok. | Output tok. |")
    assert "| gemma4-e4b | History (transcript) | reasoning off, cap 1,024 | 4 | 4 |" in md
    assert "not measured on this hardware" in md and "204 GB" in md and "accounting estimate" in md
    tex = (root / "panel_table.tex").read_text()
    assert r"\toprule" in tex and "not measured on this hardware" in tex
    rec = json.loads((root / "glm5.3-flash-nvfp4-think" / "not_measured.json").read_text())
    assert rec["measured"] is False and rec["registry_spec"]["name"] == "glm5.3-flash-nvfp4-think"


def test_server_log_parser(tmp_path):
    log = tmp_path / "s.log"
    log.write_text("INFO Model loading took 15.2 GiB and 41.3 seconds\nINFO init engine (profile, create kv cache, warmup model) took 33.5 s\n"
                   "INFO Prefix cache hit rate: 12.5%\nINFO Prefix cache hit rate: 20.0%\nINFO GPU KV cache size: 1,234,567 tokens\n")
    d = panel.parse_server_log(log)
    assert d["model_loading"]["groups"] == ["15.2", "41.3"] and d["init_engine_s"]["groups"] == ["33.5"]
    assert d["prefix_cache_hit_rate"]["last_percent"] == 20.0 and d["prefix_cache_hit_rate"]["max_percent"] == 20.0
    assert d["torch_compile_s"] is None and panel.parse_server_log(tmp_path / "none.log") is None


def test_model_identity_reads_blob_names(tmp_path):
    snap = tmp_path / "models--x" / "snapshots" / "abc"
    blobs = tmp_path / "models--x" / "blobs"
    snap.mkdir(parents=True)
    blobs.mkdir()
    (blobs / ("a" * 64)).write_bytes(b"w")
    (snap / "model.safetensors").symlink_to("../../blobs/" + "a" * 64)
    spec = m.get_spec("gemma4-e4b")
    ident = panel.model_identity(str(snap), spec)
    f = ident["files"][0]
    assert f["blob"] == "a" * 64 and f["blob_kind"] == "sha256 of content" and ident["revision_matches_registry"] is False


# ---------------------------------------------------------------------------------------------- retry of failed items

import hashlib
from evals_runner import eq1_monitor_panel_retry as retry


def sha_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir()) if p.is_file()}


@pytest.fixture
def retry_world(tmp_path):
    """4 tasks; lcbhard_0 has 450k characters (~150k tokens by the 3 chars/token estimate) of private reasoning, so its reasoning view
    is longer than a 131,072-token server context, while its transcript and final-code views are small."""
    splits, root = write_world(tmp_path, [f"lcbhard_{i}" for i in range(4)])
    big = make_traj("lcbhard_0")
    big.events[2] = tj.Event(seq=2, kind=tj.REASONING, call_index=0, data={"text": "R" * 225_000 + "MIDDLE" + "S" * 224_994})
    (root / "trajectories" / f"{prompts.rollout_id('lcbhard_0')}.json").write_text(big.to_json())
    out = tmp_path / "p.json.gz"
    prompts.main(["--trajectories", str(root), "--splits", str(splits), "--n", "4", "--out", str(out)])
    return out, tmp_path / "p.manifest.json", tmp_path, root


def original_run(retry_world, name="qwen3.8-27b-4k", plan=None):
    p, man, tmp, root = retry_world
    (tmp / "panel" / "inputs").mkdir(parents=True, exist_ok=True)
    for f in (p, man):
        (tmp / "panel" / "inputs" / f.name).write_bytes(f.read_bytes())
    stub = Stub(plan=plan, max_ctx=131072)
    rc = panel.main(["--monitor", name, "--url", stub.url, "--prompts", str(tmp / "panel/inputs" / p.name), "--manifest",
                     str(tmp / "panel/inputs" / man.name), "--out-dir", str(tmp / "panel" / name), "--allow-char-counter"])
    stub.close()
    return rc, tmp / "panel" / name


def run_retry_cmd(retry_world, stub, orig, out="retry1", extra=()):
    p, man, tmp, root = retry_world
    return panel.main(["--retry", "--panel-dir", str(orig), "--url", stub.url, "--trajectories", str(root), "--out-dir",
                       str(tmp / "panel" / out / orig.name), "--allow-char-counter", "--prompts", str(tmp / "panel/inputs" / p.name),
                       "--manifest", str(tmp / "panel/inputs" / man.name), *extra]), tmp / "panel" / out / orig.name


def test_failed_items_by_rule():
    ok = {"item": 0, "attempt": 1, "http_error": None, "valid_score": True, "seconds": 1.0}
    rows = [ok,
            {"item": 1, "not_sent": True, "http_error": None, "valid_score": False},
            {"item": 2, "attempt": 1, "http_error": "HTTPError: 400", "http_status": 400, "valid_score": False, "seconds": .1},
            {"item": 3, "attempt": 1, "http_error": None, "valid_score": False, "finish_reason": "length", "seconds": 9.0},
            {"item": 4, "attempt": 1, "http_error": None, "valid_score": False, "finish_reason": "stop", "seconds": 2.0},
            {"item": 5, "attempt": 1, "http_error": "x", "valid_score": False, "seconds": .1},
            {"item": 5, "attempt": 2, "http_error": None, "valid_score": True, "seconds": 3.0}]
    f = retry.failed_items(rows, 7)
    assert sorted(f) == [1, 2, 3, 4, 6]
    assert f[1]["reason"].startswith("not sent") and f[2]["reason"] == "http error: status 400"
    assert f[3]["reason"].startswith("length stop") and f[4]["reason"].startswith("reply without") and f[6]["reason"] == "no row for this item"


def test_budget_rule_numbers_for_the_four_monitors():
    rule = lambda n: retry.budget_rule(m.get_spec(n).config(), 131072)
    assert rule("gemma4-e4b")["fit_budget"] == 128000 and rule("gemma4-e4b")["binding_limit"] == "registry prompt budget"
    assert rule("gemma4-e4b-think")["fit_budget"] == 112000 and rule("gemma4-e4b-think")["output_allowance"] == 16384
    assert rule("qwen3.8-27b-4k")["fit_budget"] == 131072 - 4096 - 512 and rule("qwen3.8-27b-4k")["binding_limit"] == "server context"
    assert rule("qwen3.8-27b-think")["fit_budget"] == 131072 - 8192 - 512
    cfg = m.get_spec("qwen3.8-27b-think").config()
    assert retry.raised_cap(cfg, 131072, 2983) == 131072 - 2983 - 64 - 512


def test_refit_uses_the_detectors_fitting_path(tmp_path):
    traj = make_traj("lcbhard_1")
    traj.events[2] = tj.Event(seq=2, kind=tj.REASONING, call_index=0, data={"text": "a" * 90_000})
    cfg = m.get_spec("gemma4-e4b").config()
    counter = m.CharCounter(3.0)
    item = {"view": tj.COT, "input": "history"}
    res = retry.refit_view(traj, item, cfg, counter, 10_000)
    ref, info = m.build_prompt_fitted(traj, tj.COT, counter=counter, budget_tokens=10_000, overhead=cfg.template_overhead_tokens)
    assert res["ok"] and res["user"] == ref and res["reasoning_chars_cut"] == info["truncated_reasoning_chars"] > 0
    assert res["user_tokens"] <= 10_000
    rep = res["report"]
    assert rep["blocks_shortened"] == 1 and rep["changed_blocks"][0]["seq"] == 2 and 0 < rep["reasoning_fraction_kept"] < 1
    assert "truncated to fit" in res["user"]
    # the final-code view has nothing to shorten
    assert not retry.refit_view(traj, {"view": tj.BLACK_BOX, "input": "final_state"}, cfg, counter, 10)["ok"]


def test_retry_end_to_end_export_and_immutability(retry_world):
    rc, orig = original_run(retry_world)
    assert rc == 0
    rows = rows_of(orig)
    failed = {r["item"]: r for r in rows if r.get("http_error") or not r.get("valid_score") or r.get("not_sent")}
    assert 4 in failed and failed[4]["http_status"] == 400        # item 4 = lcbhard_0 reasoning view, over the stub's context
    before = sha_dir(orig)
    stub = Stub(max_ctx=131072)
    rc2, d = run_retry_cmd(retry_world, stub, orig)
    stub.close()
    assert rc2 == 0
    assert sha_dir(orig) == before                                 # the original run is untouched
    cfg = json.loads((d / "run_config.json").read_text())
    c = cfg["config"]
    assert c["kind"] == "retry" and c["original"]["rows_sha256"] == before["rows.jsonl"] and c["original"]["run_config_sha256"]
    assert c["failed_items"] == {"4": "http error: status 400"}
    rrows = rows_of(d)
    assert all(r["retry"] for r in rrows)
    r4 = [r for r in rrows if r["item"] == 4][0]
    assert r4["retry_kind"] == "refit" and r4["evidence_shortened"] is True and r4["original_user_tokens"] > 126464 >= r4["user_tokens_counted"]
    assert r4["user_sha256"] != r4["original_user_sha256"] and r4["output_cap"] == 4096 == r4["original_output_cap"] and r4["valid_score"] is True
    assert r4["http_error"] is None and r4["start_utc"] >= cfg["written_utc"]
    plan4 = [p for p in c["plan"] if p["item"] == 4][0]
    assert plan4["budget"]["fit_budget"] == 126464 and plan4["shortening"]["blocks_shortened"] == 1 and plan4["new_user_sha256"] == r4["user_sha256"]
    # the stub received the shortened prompt: it is within the context and keeps both ends of the long reasoning block
    body = [b for b in stub.bodies if "reasoning truncated to fit" in b["messages"][0]["content"]][0]
    assert "MIDDLE" not in body["messages"][0]["content"] and "R" * 1000 in body["messages"][0]["content"] and body["max_tokens"] == 4096
    gz = json.loads(gzip.decompress((d / "retry_prompts.json.gz").read_bytes()))["items"]
    assert hashlib.sha256(gz["4"].encode()).hexdigest() == r4["user_sha256"] and c["retry_prompts_sha256"] == panel.sha256_file(d / "retry_prompts.json.gz")
    s = json.loads((d / "summary.json").read_text())
    pc = s["per_condition"]["history_reasoning"]
    assert pc["final_valid"] == pc["invocations"] == 4 and pc["retried_items"] == [4] and pc["evidence_shortened_items"] == [4] and pc["still_failed_items"] == []
    assert pc["retry_kinds"] == {"refit": 1}
    # both time quantities
    e4 = [e for e in s["items"] if e["item"] == 4][0]
    orig4 = [r for r in rows if r["item"] == 4][0]["seconds"]
    assert e4["success_source"] == "retry" and e4["success_seconds"] == r4["seconds"]
    assert e4["retry_inclusive_seconds"] == pytest.approx(orig4 + r4["seconds"]) and e4["original_attempt_seconds"] == pytest.approx(orig4)
    e0 = [e for e in s["items"] if e["item"] == 0][0]
    assert e0["success_source"] == "original" and e0["retry_inclusive_seconds"] == e0["success_seconds"]
    sec = pc["successful_invocation_seconds"]
    inc = pc["retry_inclusive_seconds_of_valid_items"]
    assert inc["sum"] == pytest.approx(sec["sum"] + orig4) and inc["mean"] > sec["mean"]
    # a rerun of the same command sends nothing; a changed config is refused
    n = stub.calls
    stub2 = Stub(max_ctx=131072)
    assert run_retry_cmd(retry_world, stub2, orig)[0] == 0 and stub2.calls == 0
    with pytest.raises(SystemExit, match="refusing to resume"):
        run_retry_cmd(retry_world, stub2, orig, extra=["--max-attempts", "2"])
    stub2.close()


def test_length_stop_item_gets_the_largest_cap_and_stays_failed_without_forced_score(retry_world):
    # calls: 1 warm-up, items 0..3 are calls 2..5, item 4 (huge reasoning view) is call 6 -> 400 by the stub's context rule;
    # item 8 (final-code view, task 0) is call 10: a length stop
    rc, orig = original_run(retry_world, plan={10: "length"})
    fr = retry.failed_items(rows_of(orig), 12)
    assert sorted(fr) == [4, 8] and fr[8]["reason"].startswith("length stop")
    stub = Stub(plan={3: "noscore"}, max_ctx=131072)     # retry calls: 1 warm-up, 2 = item 4, 3 = item 8 -> still no score
    rc2, d = run_retry_cmd(retry_world, stub, orig)
    stub.close()
    rr = {r["item"]: r for r in rows_of(d)}
    cfg = m.get_spec("qwen3.8-27b-4k").config()
    ut = rr[8]["original_user_tokens"]
    assert rr[8]["retry_kind"] == "raise_cap" and rr[8]["original_output_cap"] == 4096
    assert rr[8]["output_cap"] == 131072 - ut - cfg.template_overhead_tokens - 512 and stub.bodies[2]["max_tokens"] == rr[8]["output_cap"]
    assert rr[8]["user_sha256"] == rr[8]["original_user_sha256"] and rr[8]["evidence_shortened"] is False
    assert rr[8]["valid_score"] is False                     # no forced score
    s = json.loads((d / "summary.json").read_text())["per_condition"]["final_code_only"]
    assert s["final_valid"] == 3 and s["still_failed_items"] == [8]
    # the failed item still counts in the retry-inclusive effort total, not in the successful-invocation statistics
    assert s["successful_invocation_seconds"]["sum"] < s["retry_inclusive_seconds_all_items_total"]


def test_out_dir_must_differ_and_unfittable_items_are_not_sent(retry_world):
    rc, orig = original_run(retry_world)
    stub = Stub(max_ctx=131072)
    with pytest.raises(SystemExit, match="differ from --panel-dir"):
        panel.main(["--retry", "--panel-dir", str(orig), "--url", stub.url, "--out-dir", str(orig), "--allow-char-counter"])
    # no trajectories directory: the refit item cannot be rebuilt, is recorded as not sent, and stays failed
    p, man, tmp, root = retry_world
    rc2 = panel.main(["--retry", "--panel-dir", str(orig), "--url", stub.url, "--out-dir", str(tmp / "panel" / "r_nt" / orig.name),
                      "--allow-char-counter", "--prompts", str(tmp / "panel/inputs" / p.name), "--manifest", str(tmp / "panel/inputs" / man.name)])
    stub.close()
    rr = rows_of(tmp / "panel" / "r_nt" / orig.name)
    assert rc2 == 0 and rr[0]["not_sent"] and "trajectories" in rr[0]["note"]
    s = json.loads((tmp / "panel" / "r_nt" / orig.name / "summary.json").read_text())["per_condition"]["history_reasoning"]
    assert s["final_valid"] == 3 and s["still_failed_items"] == [4]


def test_completed_table_has_no_valid_fractions_and_notes_the_changes(retry_world):
    rc, orig = original_run(retry_world)
    stub = Stub(max_ctx=131072)
    rc2, d = run_retry_cmd(retry_world, stub, orig)
    stub.close()
    root = d.parent
    assert retry.combine_retry(root, root.parent) == 0
    md = (root / "panel_table_completed.md").read_text()
    assert md.splitlines()[0] == "| Monitor | View | Config | Invocations | Seconds/invocation (mean, median) | Input tok. | Output tok. |"
    assert "Valid" not in md.splitlines()[0] and "of 4" not in md.splitlines()[2]
    assert "| qwen3.8-27b-4k | History + private reasoning | reasoning off, cap 4,096 | 4 |" in md
    assert "successful-invocation time" in md and "shortened" in md and "accounting estimate" in md and "Incomplete" not in md
    assert "\\toprule" in (root / "panel_table_completed.tex").read_text()
    summ = json.loads((root / "summary.json").read_text())
    assert summ["per_monitor"]["qwen3.8-27b-4k"]["per_condition"]["history_reasoning"]["final_valid"] == 4
    assert set(summ["time_quantities"]) == {"successful_invocation_seconds", "retry_inclusive_seconds"}
