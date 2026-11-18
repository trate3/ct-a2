import json
import urllib.error

import pytest

from evals.reward_hacking import swebench_tasks
from evals.reward_hacking.scripts import swebench_pin_images as pin
from evals.reward_hacking.scripts import swebench_preflight as pf
from evals.reward_hacking.scripts import swebench_preflight_merge as merge

D1, D2, D3 = ("sha256:" + c * 64 for c in "abc")
INDEX = json.dumps({"manifests": [
    {"digest": D3, "platform": {"os": "unknown", "architecture": "unknown"}},
    {"digest": D2, "platform": {"os": "linux", "architecture": "amd64"}}]}).encode()


def test_parse_index_picks_linux_amd64_child():
    h = {"docker-content-digest": D1, "content-type": "application/vnd.oci.image.index.v1+json"}
    assert pin.parse_manifest_response(h, INDEX) == {
        "manifest_digest": D1, "manifest_kind": "index", "platform_digest": D2}
    assert pin.parse_manifest_response(h, b"") == {"manifest_digest": D1, "manifest_kind": "index"}  # HEAD


def test_parse_single_manifest_and_errors():
    h = {"docker-content-digest": D1, "content-type": "application/vnd.docker.distribution.manifest.v2+json"}
    assert pin.parse_manifest_response(h, b"{}") == {"manifest_digest": D1, "manifest_kind": "manifest"}
    with pytest.raises(ValueError):
        pin.parse_manifest_response({**h, "docker-content-digest": "nope"}, b"")
    with pytest.raises(ValueError, match="amd64"):
        pin.parse_manifest_response({**h, "content-type": "application/vnd.oci.image.index.v1+json"},
                                    json.dumps({"manifests": []}).encode())
    with pytest.raises(ValueError, match="media type"):
        pin.parse_manifest_response({**h, "content-type": "text/html"}, b"")


def fake_fetch(table):
    def fetch(method, url, headers=None):
        if "auth.docker.io" in url:
            return {}, b'{"token": "t"}'
        for suffix, resp in table.items():
            if url.endswith(suffix):
                if resp == 404:
                    raise pin.NotFound(url)
                return resp
        raise AssertionError(url)
    return fetch


def test_pin_all_resumes_reports_missing_and_carries_v1():
    ids = ["django__django-15863", "sympy__sympy-16597", "pytest-dev__pytest-5787"]
    ok = {"docker-content-digest": D1, "content-type": "application/vnd.oci.image.index.v1+json"}
    reg = pin.Registry(fake_fetch({
        "django_1776_django-15863/manifests/latest": (ok, INDEX),
        "sympy_1776_sympy-16597/manifests/latest": 404,
        "pytest-dev_1776_pytest-5787/manifests/latest": (ok, INDEX)}))
    v1 = {"django__django-15863": {"manifest_digest": D1, "verified": True, "sif_sha256": "f" * 64},
          "pytest-dev__pytest-5787": {"manifest_digest": D3, "verified": True}}
    doc = {"version": 1, "dataset_revision": "r", "images": {
        "pytest-dev__pytest-5787": {"manifest_digest": D3, "verified": False}}}  # already pinned: skipped
    missing = pin.pin_all(ids, doc, v1, reg, log=lambda *_: None)
    assert missing == ["sympy__sympy-16597"]
    e = doc["images"]["django__django-15863"]
    assert e["image"] == "swebench/sweb.eval.x86_64.django_1776_django-15863"
    assert (e["verified"], e["sif_sha256"], e["platform_digest"]) == (True, "f" * 64, D2)
    assert doc["images"]["pytest-dev__pytest-5787"]["verified"] is False


def test_carry_over_requires_matching_digest():
    e = pin.carry_over({"manifest_digest": D1, "verified": False}, {"manifest_digest": D2, "verified": True})
    assert e["verified"] is False


def test_http_retries_429_then_succeeds(monkeypatch):
    calls = []

    class R:
        headers = {"Docker-Content-Digest": D1}
        def read(self): return b"x"
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def urlopen(req, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.HTTPError("u", 429, "slow", {"Retry-After": "7"}, None)
        return R()

    monkeypatch.setattr(pin.urllib.request, "urlopen", urlopen)
    sleeps = []
    h, b = pin.http("GET", "http://x", sleep=sleeps.append)
    assert h["docker-content-digest"] == D1 and sleeps == [7.0, 7.0]


# ---- preflight classification -------------------------------------------------------------------------------

COLLECT_TAIL = ("SWE-bench evaluation is indeterminate (script exit=2, test exit=2)\n"
                "==== ERROR collecting lib/matplotlib/tests/test_x.py ====\nNameError: name 'foo' is not defined\n"
                "!!!! Interrupted: 1 error during collection !!!!")


def test_classify_error_malformed_vs_infra():
    assert pf.classify_error(COLLECT_TAIL)[0] == "malformed"
    assert pf.classify_error("test_patch does not apply at pinned base abc; refusing")[0] == "malformed"
    assert pf.classify_error("selected SWE-bench test patch did not apply cleanly (exit 1)")[0] == "malformed"
    assert pf.classify_error("SWE-bench test command did not complete; script exit=97")[0] == "malformed"
    assert pf.classify_error("ERROR: Failed to import test module: x\n"
                             "SWE-bench evaluation is indeterminate (script exit=1, test exit=1)")[0] == "malformed"
    # collection words outside the scorer's indeterminate error do not count; neither do other failures
    assert pf.classify_error("Apptainer pull failed: ERROR collecting layers")[0] == "infra"
    assert pf.classify_error("SWE-bench test command did not complete; script exit=124")[0] == "infra"
    assert pf.classify_error("SWE image x lacks required base commit y")[0] == "infra"
    assert pf.classify_error("SWE-bench evaluation is indeterminate (script exit=1, test exit=1)\nConnectionError")[0] == "infra"


def test_classify_control():
    assert pf.classify_control(kind="test_failure", value=0.0) == ("ok", "test_failure")
    assert pf.classify_control(kind="pass", value=1.0)[0] == "infra"
    assert pf.classify_control(kind="flagged", value=0.0)[0] == "infra"
    assert pf.classify_control(kind=None, error=COLLECT_TAIL)[0] == "malformed"
    assert pf.classify_control(kind="test_failure", error="boom")[0] == "infra"


def test_order_and_shards():
    parts = {"train": [{"instance_id": "b"}, {"instance_id": "a"}], "val": [{"instance_id": "c"}],
             "test": [{"instance_id": "d"}, {"instance_id": "e"}]}
    items = pf.order_ids(parts)
    assert [i for i, _ in items] == ["a", "b", "c", "d", "e"]
    assert pf.shard_items(items, "1/2") == [("b", "train"), ("d", "test")]
    assert sorted(x for s in range(3) for x in pf.shard_items(items, f"{s}/3")) == sorted(items)
    with pytest.raises(ValueError):
        pf.shard_items(items, "2/2")


def test_real_manifest_shards_partition_all_instances():
    from evals.reward_hacking.scripts.swebench_pin_images import all_instance_ids
    items = pf.order_ids(swebench_tasks.load_manifest()["partitions"])
    assert len(items) == len(all_instance_ids()) == 349
    assert items[0][1] == "train" and items[-1][1] == "test"
    assert sum(len(pf.shard_items(items, f"{i}/8")) for i in range(8)) == 349


def test_done_ids_resume(tmp_path):
    (tmp_path / "results-0of2.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"id": "a", "status": "ok"}, {"id": "b", "status": "infra"}, {"id": "c", "status": "malformed"}]) + "\n{bad")
    assert pf.done_ids(tmp_path) == {"a", "c"}
    assert pf.done_ids(tmp_path, retry_infra=False) == {"a", "b", "c"}


def test_preflight_one_with_stub(tmp_path, monkeypatch):
    sif = tmp_path / "x.sif"
    sif.write_bytes(b"sif")
    monkeypatch.setattr(pf, "check_base_commit", lambda *a: (True, ""))
    entry = {"image": "swebench/i", "manifest_digest": D1}
    seen = []

    def control(iid, variant, dummy, digest, log):
        seen.append((variant, dummy))
        return {"kind": "test_failure" if variant == "conflicting" else "pass", "value": 0.0, "error": None}

    rec = pf.preflight_one("x", "train", entry, "bc", images=tmp_path, out=tmp_path, apptainer="a",
                           original_oracle=True, clean_root=False, control=control)
    assert rec["status"] == "ok" and rec["pulled"] is False and rec["size_bytes"] == 3
    assert rec["sif_sha256"] == pf.file_sha256(sif) and rec["original_oracle"]["pass"] is True
    assert seen == [("conflicting", "nochange"), ("original", "oracle")]
    bad = pf.preflight_one("x", "train", {**entry, "sif_sha256": "0" * 64}, "bc", images=tmp_path, out=tmp_path,
                           apptainer="a", original_oracle=False, clean_root=False, control=control)
    assert bad["status"] == "infra" and "pinned" in bad["error_tail"]


# ---- merge ---------------------------------------------------------------------------------------------------

def test_merge(tmp_path):
    pins = {"version": 1, "dataset_revision": "r", "images": {
        k: {"image": "i", "manifest_digest": D1, "verified": False} for k in "abcd"}}
    recs = {"a": {"id": "a", "status": "ok", "sif_sha256": "1" * 64},
            "b": {"id": "b", "status": "malformed", "partition": "val", "reason": "r", "error_tail": "ERROR collecting"},
            "c": {"id": "c", "status": "infra"}}
    pins, mal, summary = merge.merge(pins, recs)
    assert pins["images"]["a"]["verified"] is True and pins["images"]["a"]["sif_sha256"] == "1" * 64
    assert pins["images"]["b"]["verified"] is False and "c" not in mal["ids"]
    assert mal["ids"] == ["b"] and mal["details"]["b"]["partition"] == "val"
    assert summary == {"ok": ["a"], "malformed": ["b"], "infra": ["c"], "not_run": ["d"]}
    pins["images"]["a"]["sif_sha256"] = "2" * 64
    with pytest.raises(ValueError):
        merge.merge(pins, recs)


def test_latest_records_keeps_final_verdict(tmp_path):
    (tmp_path / "results-0of2.jsonl").write_text(
        json.dumps({"id": "a", "status": "ok", "ts": "1"}) + "\n" + json.dumps({"id": "b", "status": "infra", "ts": "1"}) + "\n")
    (tmp_path / "results-1of2.jsonl").write_text(
        json.dumps({"id": "a", "status": "infra", "ts": "2"}) + "\n" + json.dumps({"id": "b", "status": "ok", "ts": "2"}) + "\n")
    r = merge.latest_records(tmp_path)
    assert r["a"]["status"] == "ok" and r["b"]["status"] == "ok"


def test_classify_error_broken_module_is_malformed():
    from evals.reward_hacking.scripts.swebench_preflight import classify_error
    base = "SWE-bench evaluation is indeterminate (script exit=1, test exit=1)\n"
    for tail in ('  File "/testbed/tests/x.py", line 370, in BaseTests\n    self.assertEqual(a, b)\n'
                 "NameError: name 'self' is not defined",
                 '  File "/testbed/tests/migrations/test_operations.py", line 2994\nIndentationError: unexpected indent'):
        assert classify_error(base + tail)[0] == "malformed"
    assert classify_error(base + "Ran 21 tests\nFAILED (errors=1)")[0] == "infra"
