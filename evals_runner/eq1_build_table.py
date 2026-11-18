"""Build the EQ1 phase table from the output folders of a GPU run and print it with reproduce/eq1_table.py.

It checks the run outputs (monitor configurations, input and instruction hashes, every saved reply re-parsed, first-attempt
valid rows, means recomputed against the saved summaries, the attestation mode and GPU policy of the certification timings),
writes the numeric records into a folder laid out like data/eq1, and prints the table from that folder.

  python -m evals_runner.eq1_build_table --gen-dir OUT --inputs OUT/judges/inputs/eq1_judge_saved_calls_inputs.json.gz \
      --monitors OUT/judges --tap-work OUT/tap_work/summary.json --cert-dir OUT/certification/run --out-dir OUT/table
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from reproduce import eq1_table

ROOT = Path(__file__).resolve().parents[1]
ATTESTATION_MODE = "nvidia-gpu-only-software-signed"
MONITORS = tuple(name for name, _, _ in eq1_table.MONITORS)
CERT_CONFIG = {name: config for name, _, config in eq1_table.MONITORS}
CALL_KEYS = ("history_id", "task", "depth", "run_key", "arm", "seed", "prompt_tokens", "max_new_tokens", "status",
             "elapsed_s", "n_generated", "stopped_naturally", "finish_reason", "token_hash", "tap_score_s")
ROW_KEYS = ("item", "run_key", "history_id", "task", "attempt", "seconds", "http_status", "finish_reason", "valid_score",
            "prompt_tokens", "generated_tokens")


def read(p): return json.loads(Path(p).read_text())
def jl(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def objsha(o): return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def wjl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def wj(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1) + '\n')


def equal(a, b):
    assert np.allclose([a['estimate'], *a['ci95']], [b['estimate'], *b['ci95']], atol=1e-8, rtol=0), (a, b)


def check_certification(cert: dict, policy_digest: str | None, cc_text: Path | None) -> None:
    """The timings must come from the GPU attestation mode, under the given policy digest."""
    assert cert['attestation_mode'] == ATTESTATION_MODE, f"certification timings were taken with attestation mode {cert['attestation_mode']!r}, not {ATTESTATION_MODE}"
    assert cert.get('policy_enforced') is True and cert.get('policy_digest'), 'the GPU policy was not enforced in the certification timings'
    if policy_digest:
        assert cert['policy_digest'] == policy_digest, 'certification timings were taken under another GPU policy'
    assert cert.get('gpu_attestation_verified') is True and cert.get('workload_attested') is False and cert.get('full_certification') is False
    if cc_text:
        txt = cc_text.read_text()
        assert 'CC State' in txt and ': ON' in txt.split('CC State', 1)[1].splitlines()[0], 'conf-compute -q does not show CC State ON'


def check_monitor(d: Path, name: str, gen: Path, stage2_list: Path, sel: dict, inputs: list, gpu_key: str | None, code_ref: str | None) -> tuple[dict, dict]:
    """Verify one monitor folder; return its per-prompt judging seconds and its record for verification.json."""
    from evals.reward_hacking import monitors as m
    from evals_runner.eq1_monitor_panel import reply_valid

    positions = sel['positions']
    rec = read(d / 'run_config.json'); cfg = rec['config']
    assert objsha(cfg) == rec['config_sha256']
    assert cfg['outputs_sha256'] == sha(gen / 'outputs.jsonl')
    assert cfg['stage2_list_sha256'] == sha(stage2_list)
    assert cfg['instruction'] == 'early'
    assert cfg['instruction_sha256'] == hashlib.sha256(m.EARLY_INSTRUCTION.encode()).hexdigest()
    assert cfg['histories_sha256'] == sel['histories_sha256']
    assert cfg['positions'] == [{k: p[k] for k in ('history_id', 'task', 'weight', 'run_key')} for p in positions]
    gpu_name = cfg['hardware']['nvidia_smi_gpu_name_memory_total_mib_driver_power_limit_w']
    if gpu_key:
        assert gpu_key in gpu_name, (gpu_key, gpu_name)
    if code_ref is not None:
        assert cfg['code_sha256'] == code_ref, 'monitor code differs from the reference run'
    raw = jl(d / 'rows.jsonl'); replies = {(r['item'], r['attempt']): r['text'] for r in jl(d / 'replies.jsonl')}
    assert len(raw) == 24 and len({r['item'] for r in raw}) == 24, f'{name}: need 24 first-attempt rows'
    vals = {}
    for r in raw:
        assert r['valid_score'] and r['finish_reason'] == 'stop' and r['attempt'] == 1 and r['http_status'] == 200, f"{name} item {r['item']}"
        assert not r['fitted'] and not r['evidence_shortened']
        assert r['user_sha256'] == r['original_user_sha256'] == inputs[r['item']]['user_sha256']
        assert r['run_key'] == inputs[r['item']]['run_key']
        assert rec['written_utc'] < r['start_utc']
        valid, why = reply_valid(replies[r['item'], r['attempt']], r['finish_reason'], m.MonitorConfig(**cfg['monitor_config']))
        assert valid and not why
        vals[r['run_key']] = r['seconds']
    return vals, {'monitor': name, 'config_sha256': rec['config_sha256'], 'rows_sha256': sha(d / 'rows.jsonl'),
                  'replies_sha256': sha(d / 'replies.jsonl'), 'code_sha256': cfg['code_sha256'], 'gpu': gpu_name}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gen-dir', type=Path, required=True, help='folder with generate_calls_stage2/{calls,setup,outputs}.jsonl')
    ap.add_argument('--stage2-list', type=Path, default=ROOT / 'data/eq1/generation/stage2_list.json')
    ap.add_argument('--main-complete-call', type=Path, default=None, help='main_complete_call.json; built from the stage-2 calls when absent')
    ap.add_argument('--inputs', type=Path, required=True, help='eq1_judge_saved_calls_inputs.json.gz of the monitor runs')
    ap.add_argument('--monitors', type=Path, required=True, help='folder with <NAME>/run_config.json for each monitor')
    ap.add_argument('--gpu-name-contains', default=None, help='text that must appear in the GPU name recorded by every monitor run')
    ap.add_argument('--code-sha-reference', type=Path, default=None, help='a run_config.json whose code_sha256 every monitor run must equal')
    ap.add_argument('--tap-work', type=Path, required=True, help='summary.json of evals.tap_live.profile_components')
    ap.add_argument('--cert-dir', type=Path, required=True, help='output folder of evals_runner.eq1_cert_timing (reps.jsonl, cert_timings.json)')
    ap.add_argument('--policy-digest', default=os.environ.get('EQ1_GPU_POLICY_DIGEST'),
                    help='the GPU policy digest the certification timings must have been taken under [EQ1_GPU_POLICY_DIGEST]')
    ap.add_argument('--require-cc-text', type=Path, default=None, help='file with `nvidia-smi conf-compute -q` output; must show CC State ON')
    ap.add_argument('--out-dir', type=Path, required=True, help='written like data/eq1, then printed')
    ap.add_argument('--json', type=Path, default=None, help='also write the rows with intervals to this file')
    a = ap.parse_args(argv)

    gen = a.gen_dir / 'generate_calls_stage2'
    sel = read(a.stage2_list)
    positions = sel['positions']
    stat = eq1_table.make_stat(positions)
    calls = jl(gen / 'calls.jsonl')
    a0 = {r['run_key']: r['elapsed_s'] for r in calls if r['arm'] == 'A0'}
    setup = jl(gen / 'setup.jsonl')[0]
    with gzip.open(a.inputs, 'rt') as f:
        inputs = json.load(f)['items']

    if a.main_complete_call:
        main_doc, main_sha = read(a.main_complete_call), sha(a.main_complete_call)
    else:
        from evals.tap_live import main_complete_call as mcc
        main_doc, main_sha = mcc.build(None, gen, sel), None
        assert main_doc['positions_by_source'].get('stage1_r0', 0) == 0, 'the main stage uses stage 1 positions'
    assert main_doc['complete'], f"main_complete_call lists missing positions: {main_doc['missing_positions']}"
    equal(stat(a0), main_doc['primary']['no_tap_mean_s_per_call'])      # guards against a stale main_complete_call.json

    code_ref = read(a.code_sha_reference)['config']['code_sha256'] if a.code_sha_reference else None
    verified, startup, monitor_rows = [], {}, {}
    for name in MONITORS:
        d = a.monitors / name
        vals, record = check_monitor(d, name, gen, a.stage2_list, sel, inputs, a.gpu_name_contains, code_ref)
        code_ref = code_ref or record['code_sha256']             # all monitor runs share one code version
        summary = read(d / 'summary.json')['views']['history_black_box']
        equal(stat(vals), summary['tap_s_per_call'])
        equal(stat({k: a0[k] + v for k, v in vals.items()}), summary['composed']['with_tap_s_per_call'])
        if main_sha:
            assert summary['composed']['main_complete_call_sha256'] == main_sha
        startup[name] = jl(d / 'sessions.jsonl')[0]['server_info_session']['seconds_launch_to_health_ok']
        monitor_rows[name] = jl(d / 'rows.jsonl')
        verified.append(record)

    cert = read(a.cert_dir / 'cert_timings.json')
    check_certification(cert, a.policy_digest, a.require_cc_text)
    reps = jl(a.cert_dir / 'reps.jsonl')
    for config in {'probe', *CERT_CONFIG.values()}:
        assert any(r['config'] == config and r['cache_state'] == 'warm' for r in reps), f'no repeated certification session for configuration {config!r}'

    out = a.out_dir
    if out.exists():
        shutil.rmtree(out)
    wjl(out / 'generation/calls.jsonl', [{k: r[k] for k in CALL_KEYS} for r in calls])
    wj(out / 'generation/setup.json', {'model_load_s': setup['model_load_s'], 'artifact_load_s': setup['artifact_load_s']})
    shutil.copyfile(a.stage2_list, out / 'generation/stage2_list.json')
    for name, rows in monitor_rows.items():
        wjl(out / 'monitors' / name / 'rows.jsonl', [{k: r[k] for k in ROW_KEYS} for r in rows])
    wj(out / 'monitors/startup.json', {'seconds_launch_to_health_ok': startup})
    wjl(out / 'certification/attestation.jsonl',
        [{'config': r['config'], 'attestation_fetch_s': r['attestation_fetch_s'], 'attestation_verify_s': r['attestation_verify_s']} for r in reps])
    wjl(out / 'certification/sessions.jsonl', [{'config': r['config'], 'certification_s': r['protocol_overhead_s']} for r in reps if r['cache_state'] == 'warm'])
    prof = read(a.tap_work)
    metrics = prof['metrics']
    wj(out / 'tap_work.json', {
        'unit': 'ms per call, weighted over the 30 positions, 95% task-bootstrap interval',
        'tap_work_ms': metrics['alone_net_ms']['weighted_mean_over_positions'],
        'hook_calls_ms': metrics['hook_cpu_net_ms']['weighted_mean_over_positions'],
        'end_of_call_scoring_ms': metrics['end_call_ms']['weighted_mean_over_positions'],
        'weighted_mean_generated_tokens': prof['weighted_mean_generated_tokens']})
    wj(out / 'verification.json', {'monitors': verified, 'main_stage_run_config_sha256': read(gen / 'run_config.json').get('config_sha256'),
                                   'main_complete_call_sha256': main_sha, 'cert_timings_sha256': sha(a.cert_dir / 'cert_timings.json'),
                                   'policy_digest': cert['policy_digest'], 'key_cache_policy': cert.get('key_cache_policy')})
    return eq1_table.main(['--data', str(out)] + (['--json', str(a.json)] if a.json else []))


if __name__ == '__main__':
    raise SystemExit(main())
