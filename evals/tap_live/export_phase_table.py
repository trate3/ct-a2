"""Export the decode_bench results as the project's EQ1 phase-table row (live measurements only).

  python -m evals.tap_live.export_phase_table --decode-dir OUT [--length-dir OUT_LENGTH] --out-dir DIR \
      [--template internal notes]

Unit: one inference = one model generation call (a prefilled history, capture during generation, scoring at call end), capped at
the token budget; NOT one generated token. "No tap (s)" and "With tap (s)" are MEAN elapsed seconds per timed call over the same
paired runs (all-attempt pairs: both arms have an elapsed time); "Tap (ms/inf.)" is the mean paired (A1 - A0) elapsed difference in
milliseconds per call over those pairs (= With tap minus No tap of the same pairs: ONE pair-weighted estimand for the means, the
difference and the interval), with a 95% bootstrap interval that resamples whole TASKS (2,000 draws, seed 0). The equal-weight-per-
history estimand, the relative overhead (a fraction) and the normalized cost per 100 generated tokens are separate JSON entries;
the clean subset is in the JSON. length_check rows carry the raw paired range, not an interval (three histories). One combined row for the probes attached in one arm (the row says which artifacts the arm carries). Setup (s) =
measured model + tap-artifact load time of the decode_bench process (`setup.jsonl`), outside the timed region; the warm-up is
reported separately. Attestation and Total certification are "--": not measured, no stubs, no sums of unrelated experiments.
Block rows from `length_check` (1,024 and 4,096 tokens) are separate rows; their Setup is "--" because that script does not measure
loading. The project's table skeleton is kept verbatim; only the rows between \\midrule and the trailing comments are replaced.
Outputs: eq1_phase_row.json, eq1_phase_rows.tex (the row lines) and eq1_phase_table.tex (the filled table).
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from evals.tap_live import summarize_decode as sd

DEFAULT_TEMPLATE = Path(__file__).resolve().parents[2] / "review_exchange" / "eq1_phases_live_table.tex"
POOLING_TEXT = {"token_mean": "mean", "ema_max": "EMA-max"}
CAPTION_NOTE = ("The evaluation columns are the generation block, not a complete agent turn: up to {budget} newly generated tokens per "
                "call with natural stops kept (mean {g0:.0f} generated tokens without the tap and {g1:.0f} with it; natural stops "
                "{n0} and {n1} of {n} calls); the prefill of the historical prefix is excluded, and a capped block that ends before the "
                "turn would have finished is a generation-block measurement at that cap, not the cost of a complete agent turn. "
                "No tap and With tap are mean elapsed seconds per generation call over the same {n_pairs} paired timed calls "
                "({n_tasks} tasks, {n_hist} task-depth histories), pair-weighted; Tap (ms/inf.) is the mean paired difference per call "
                "(With tap minus No tap) with a 95% bootstrap interval that resamples whole tasks. {interval_sentence}{range_sentence}")


def _mean(v):
    return statistics.fmean(v) if v else None


def paired_row(pairs: list[tuple[dict, dict]], population_interval: bool = True) -> dict | None:
    """ONE declared weighting for the displayed numbers: pair-weighted over all pairs in `pairs`. The no-tap mean, the with-tap
    mean, their difference (exactly with-tap minus no-tap of the same pairs) and the interval (tasks resampled, 2,000 draws, seed 0)
    all use it. The equal-weight-per-history estimand is shown separately in the JSON. With `population_interval=False`
    (length_check: three histories, few tasks) only the raw paired range is given, no interval."""
    if not pairs:
        return None
    no_tap, with_tap = _mean([a["elapsed_s"] for a, _ in pairs]), _mean([b["elapsed_s"] for _, b in pairs])
    diffs_ms = [1000.0 * (b["elapsed_s"] - a["elapsed_s"]) for a, b in pairs]
    out = {"n_pairs": len(pairs), "n_tasks": len({sd.task_of(a) for a, _ in pairs}), "n_histories": len({a["history_id"] for a, _ in pairs}),
           "weighting": "pair-weighted over all pairs in this row (each attempted pair counts once)",
           "no_tap_s_mean": no_tap, "with_tap_s_mean": with_tap,
           "tap_ms_per_call_mean_paired": 1000.0 * (with_tap - no_tap),
           "paired_diff_ms_raw_range": [min(diffs_ms), max(diffs_ms)],
           "mean_generated_per_call": {"A0": _mean([a["n_generated"] for a, _ in pairs]), "A1": _mean([b["n_generated"] for _, b in pairs])},
           "natural_stops": {"A0": sum(bool(a["stopped_naturally"]) for a, _ in pairs), "A1": sum(bool(b["stopped_naturally"]) for _, b in pairs)},
           "identical_token_streams": sum(a["token_hash"] == b["token_hash"] for a, b in pairs),
           "normalized_cost_per_100_tokens": sd.normalized_per_100_tokens(pairs)}
    if population_interval:
        est = sd.abs_estimand(pairs, "pair")
        out["estimand_pair_weighted_abs_ms"] = est                          # quantity: absolute milliseconds per call
        out["ci95_ms"] = est["ci95_ms"]
        out["interval_verdict"] = est["verdict"]
        lo, hi = est["ci95_ms"]
        out["interval_statement"] = ("interval includes zero (unresolved); a negative point estimate is not a speedup finding"
                                     if lo <= 0 <= hi else "interval excludes zero")
        out["estimand_history_weighted_abs_ms"] = sd.abs_estimand(pairs, "history")   # a different estimand when counts are unequal
        out["relative_overhead_fraction"] = sd.rel_estimand(pairs)                  # quantity: a fraction of baseline elapsed time
    else:
        out["ci95_ms"] = None
        out["interval_note"] = (f"no population interval: {out['n_tasks']} unique task(s) and {out['n_histories']} histories (counted from the "
                                "history ids); the raw paired range is given instead (not an interval; few tasks)")
    return out


def make_pairs(rows: list[dict], budget: int | None = None):
    by = defaultdict(dict)
    for r in rows:
        if budget is None or r.get("budget") == budget:
            by[(r["history_id"], r["repeat"])][r["arm"]] = r
    both = [(v["A0"], v["A1"]) for v in by.values() if "A0" in v and "A1" in v]
    timed = [(a, b) for a, b in both if a.get("elapsed_s") is not None and b.get("elapsed_s") is not None]
    clean = [(a, b) for a, b in timed if a["status"] == "ok" and b["status"] == "ok"]
    return both, timed, clean


def probe_cell(heads: dict) -> tuple[str, str]:
    names = [POOLING_TEXT.get(h["pooling"], h["pooling"]) for h in heads.values()]
    if len(names) == 1:
        return f"Probe: {names[0]}", f"Probe: {names[0]}"
    return f"Probes: {names[0]}\\\\and {names[1]}", f"Probes: {names[0]} and {names[1]}"


def block_row(rows, budget, heads, setup, cfg_note, population_interval=True, coverage=None) -> dict:
    both, timed, clean = make_pairs(rows, budget)
    return {"model": "Qwen3.8-27B", "task": "LiveCodeBench", "tap": probe_cell(heads)[1],
            "tap_artifacts": {n: {"sha256": h["sha256"], "layer": h["layer"], "pooling": h["pooling"], "threshold": h["threshold"]} for n, h in heads.items()},
            "unit": "one inference = one model generation call (prefilled history; capture during generation, scoring at call end)",
            "aggregation": "mean elapsed seconds per timed call over the same paired runs",
            "block": "generation block: up to the declared token budget, natural stops kept; historical-prefix prefill excluded; "
                     "not the cost of a complete agent turn",
            "block_budget_tokens": budget, "pairs_scheduled_attempted_both_arms": len(both), "setup": setup,
            "selection_coverage": coverage,
            "all_attempt": paired_row(timed, population_interval), "clean": paired_row(clean, population_interval),
            "attestation_s": None, "total_certification_s": None,
            "not_measured": "attestation and total certification (no stubs, no sums of unrelated experiments)", "source": cfg_note}


def cell_num(x: float) -> str:
    """Signed two decimals for the table cell ('-0.06', '+0.14'); significant figures when that would print 0.00."""
    return sd.fmt_sig(x, True) if x != 0 and abs(x) < 0.005 else f"{x:+.2f}"


def fmt_num(x, nd=2) -> str:
    return "--" if x is None else f"{x:.{nd}f}"


def latex_row(r: dict, tap_cell: str) -> str:
    a = r["all_attempt"]
    b = f"{r['block_budget_tokens']:,}".replace(",", "{,}")
    if a is None:
        task = f"\\shortstack[l]{{LiveCodeBench\\\\{b}-token budget}}"
    else:
        g, n = a["mean_generated_per_call"]["A0"], a["natural_stops"]["A0"]
        task = (f"\\shortstack[l]{{LiveCodeBench\\\\{b}-token budget\\\\{{\\scriptsize mean {g:.0f} tok., {n}/{a['n_pairs']} natural stops, "
                f"{a['n_tasks']} task{'s' if a['n_tasks'] != 1 else ''}}}}}")
    setup = fmt_num(r["setup"]["setup_s"]) if r["setup"] else "--"
    if a is None:
        cells = ["--", "--", "--"]
    else:
        interval = a["ci95_ms"] is not None
        lo, hi = a["ci95_ms"] if interval else a["paired_diff_ms_raw_range"]
        inner = (f"[{cell_num(lo)}, {cell_num(hi)}]" if interval else f"range [{cell_num(lo)}, {cell_num(hi)}]$^{{\\dagger}}$")
        cells = [fmt_num(a["no_tap_s_mean"], 3), fmt_num(a["with_tap_s_mean"], 3),
                 f"\\shortstack{{{cell_num(a['tap_ms_per_call_mean_paired'])}\\\\{{\\scriptsize {inner}}}}}"]
    return (f"{r['model']} & {task} & \\shortstack[l]{{{tap_cell}}}\n    & {setup} & {cells[0]} & {cells[1]} & {cells[2]} & -- & -- \\\\")


def setup_times(decode_dir: Path) -> dict | None:
    p = decode_dir / "setup.jsonl"
    if not p.exists():
        return None
    ss = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    first = ss[0]
    return {"session": first["session"], "model_load_s": first["model_load_s"], "artifact_load_s": first["artifact_load_s"],
            "setup_s": first["model_load_s"] + first["artifact_load_s"], "warmup_s_reported_separately": first.get("warmup_s"),
            "n_sessions": len(ss), "other_sessions": [{k: x[k] for k in ("session", "model_load_s", "artifact_load_s")} for x in ss[1:]],
            "note": "measured in the first decode_bench process; load time of model + tokenizer and of the tap artifact, outside the timed region"}


def build(decode_dir: Path, length_dir: Path | None = None) -> tuple[dict, list[tuple[dict, str]]]:
    cfg = json.loads((decode_dir / "run_config.json").read_text())
    c = cfg["config"]
    heads = c["artifact"]["heads"]
    rows = [json.loads(x) for x in (decode_dir / "decode.jsonl").read_text().splitlines() if x.strip()]
    hs = c.get("histories_summary") or {}
    coverage = {"unique_tasks": hs.get("n_tasks", len({h.rsplit("@", 1)[0] for h in c["history_ids"]})),
                "unique_histories": hs.get("n_histories", len(set(c["history_ids"]))),
                "earlier_call_fallbacks": hs.get("fallback_history_ids"), "n_earlier_call_fallbacks": hs.get("n_fallback_to_earlier_call")}
    out_rows = [block_row(rows, c["max_new_tokens"], heads, setup_times(decode_dir), f"decode_bench {decode_dir}", True, coverage)]
    a = out_rows[0]["all_attempt"]
    if a is not None:
        out_rows[0]["caption_note"] = CAPTION_NOTE.format(
            n_hist=a["n_histories"], n_pairs=a["n_pairs"], n_tasks=a["n_tasks"], budget=c["max_new_tokens"], n=a["n_pairs"],
            g0=a["mean_generated_per_call"]["A0"], g1=a["mean_generated_per_call"]["A1"],
            n0=a["natural_stops"]["A0"], n1=a["natural_stops"]["A1"],
            interval_sentence=(("The interval includes zero (unresolved); a negative point estimate is not a speedup finding. "
                                if a["ci95_ms"][0] <= 0 <= a["ci95_ms"][1] else "The interval excludes zero. ")),
            range_sentence="")
    if length_dir is not None and (length_dir / "length.jsonl").exists():
        lrows = [json.loads(x) for x in (length_dir / "length.jsonl").read_text().splitlines() if x.strip()]
        for b in sorted({r["budget"] for r in lrows}):
            if b == c["max_new_tokens"]:
                continue                                 # the primary block is the decode_bench row; not merged
            row = block_row(lrows, b, heads, None, f"length_check {length_dir}", False, coverage)
            row["setup_note"] = "length_check does not measure loading"
            out_rows.append(row)
        lens = [r for r in out_rows[1:] if r["all_attempt"] is not None]
        if lens and out_rows[0].get("caption_note"):
            k = sorted({r["all_attempt"]["n_tasks"] for r in lens})
            out_rows[0]["caption_note"] += (f" $^{{\\dagger}}$ The length-check rows ({', '.join(str(r['block_budget_tokens']) for r in lens)} tokens) "
                                            f"show the raw paired range (not an interval; few tasks: {', '.join(map(str, k))} unique task(s), counted from the "
                                            "history ids), not the task-bootstrap interval of the main row.")
    cells = [(r, probe_cell(heads)[0]) for r in out_rows]
    doc = {"unit": out_rows[0]["unit"], "aggregation": out_rows[0]["aggregation"], "run_config_sha256": cfg["config_sha256"],
           "rows": out_rows, "attestation_and_total_certification": "not measured (--)"}
    return doc, cells


def _tex_int(x: float) -> str:
    n = f"{x:,.0f}"
    return n.replace(",", "{,}")


def caption_latex(doc: dict) -> str:
    """Sentences added to the project's caption, generated from the row JSON only (LaTeX-safe: percent signs escaped)."""
    r0 = doc["rows"][0]
    a = r0["all_attempt"]
    if a is None:
        return ""
    longer = [r for r in doc["rows"][1:] if r["all_attempt"] is not None]
    s = []
    if r0["setup"]:
        st = r0["setup"]
        s.append(f"Setup is the measured local load of the model and tokenizer ({st['model_load_s']:.2f} s) and of the probe artifact "
                 f"({sd.fmt_sig(st['artifact_load_s'])} s), outside the recurring window (file-cache state was not controlled)"
                 + (f"; a separate warmup of {st['warmup_s_reported_separately']:.2f} s is excluded." if st.get("warmup_s_reported_separately") is not None else "."))
    s.append("Evaluation columns are mean seconds per live generation request over the same paired runs without and with the probe; the tap captures "
             "states during generation and scores them at the call boundary. They are generation blocks, not complete agent turns: up to the token "
             "budget with natural stops kept, the prefill of the historical prefix excluded, and a block capped before the turn would have finished "
             "measures the generation block at that cap.")
    s.append(f"The {r0['block_budget_tokens']}-token row uses {a['n_pairs']} paired calls from {a['n_histories']} task-depth histories on "
             f"{a['n_tasks']} tasks; its brackets give a 95\\% bootstrap interval that resamples whole tasks.")
    if longer:
        ranges = "; ".join(f"{_tex_int(r['block_budget_tokens'])} tokens: {r['all_attempt']['n_pairs']} pairs from "
                           f"{r['all_attempt']['n_tasks']} unique task{'s' if r['all_attempt']['n_tasks'] != 1 else ''}" for r in longer)
        s.append(f"The longer-budget rows ({ranges}; tasks counted from the history ids) show the observed paired range$^{{\\dagger}}$ in brackets.")
    rows = [r0] + longer
    lens = [_tex_int(r["all_attempt"]["mean_generated_per_call"]["A0"]) for r in rows]
    s.append("Mean generated lengths are " + (", ".join(lens[:-1]) + " and " + lens[-1] if len(lens) > 1 else lens[0]) + " tokens (natural stops "
             + ", ".join(f"{r['all_attempt']['natural_stops']['A0']}/{r['all_attempt']['n_pairs']}" for r in rows) + ").")
    if a.get("interval_statement", "").startswith("interval includes zero"):
        s.append("For the primary row the interval includes zero (unresolved); a negative point estimate is not evidence of a speedup.")
    elif a.get("interval_statement"):
        s.append("For the primary row the interval excludes zero.")
    s.append("Attestation and total certification were not measured.")
    if longer:
        s.append("$^{\\dagger}$ Raw paired range, not an interval (few tasks).")
    return " ".join(s)


def insert_into_caption(template: str, text: str) -> str:
    """Append `text` as additional sentences just before the closing brace of \\caption{...}; the project's wording is kept as is."""
    start = template.index("\\caption{") + len("\\caption{")
    depth, i = 1, start
    while depth:
        c = template[i]
        if c == "\\":
            i += 2
            continue
        depth += c == "{"
        depth -= c == "}"
        i += 1
    close = i - 1
    return template[:close] + "\n" + text + template[close:]


def fill_template(template: str, row_lines: list[str], caption_add: str | None = None) -> str:
    """Replace the data rows between \\midrule and the first following comment or \\bottomrule and, if given, append `caption_add` to the
    caption (before its closing brace); everything else, the label included, verbatim."""
    lines = template.splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if l.strip() == "\\midrule") + 1
    end = next(i for i in range(start, len(lines)) if lines[i].lstrip().startswith("%") or lines[i].strip() == "\\bottomrule")
    out = "".join(lines[:start]) + "".join(l + "\n" for l in row_lines) + "".join(lines[end:])
    return insert_into_caption(out, caption_add) if caption_add else out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decode-dir", type=Path, required=True)
    ap.add_argument("--length-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    a = ap.parse_args(argv)
    doc, cells = build(a.decode_dir, a.length_dir)
    doc["caption_latex"] = caption_latex(doc)
    lines = [latex_row(r, tap) for r, tap in cells]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "eq1_phase_row.json").write_text(json.dumps(doc, indent=1) + "\n")
    (a.out_dir / "eq1_phase_rows.tex").write_text("\n".join(lines) + "\n")
    (a.out_dir / "eq1_phase_table.tex").write_text(fill_template(a.template.read_text(), lines, doc["caption_latex"]))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
