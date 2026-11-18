"""Pin Docker Hub manifest digests for every Impossible-SWE-bench conflicting instance (all partitions).

  python -m evals.reward_hacking.scripts.swebench_pin_images [--out PATH] [--no-platform]

Read-only registry calls (auth.docker.io token, registry-1.docker.io manifest HEAD/GET). The pinned digest is the
digest of the tag's top-level manifest: if that is an image index (manifest list), `apptainer pull docker://img@digest`
resolves linux/amd64 from it, so we also record the linux/amd64 child digest (`platform_digest`); if it is a single
image manifest it is pulled as is. `manifest_kind` records which. Entries are written `verified: false`; the preflight
merge sets them true. Resumable: ids already pinned in --out are skipped, and the file is rewritten after each batch.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Callable

from evals.reward_hacking import swebench_tasks as splits

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "swebench"
V1 = DATA / "image_digests_v1.json"
V2 = DATA / "image_digests_v2.json"
INDEX_TYPES = {"application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json"}
MANIFEST_TYPES = {"application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json"}
ACCEPT = ", ".join(sorted(INDEX_TYPES | MANIFEST_TYPES))
DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
REGISTRY = "https://registry-1.docker.io/v2"


class NotFound(Exception):
    pass


def http(method: str, url: str, headers: dict | None = None, *, retries: int = 6, sleep=time.sleep):
    """Return (status-200 headers lowercased, body bytes). 404 -> NotFound; 429/5xx/network -> backoff retry."""
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return {k.lower(): v for k, v in r.headers.items()}, r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise NotFound(url) from e
            if e.code in (401, 429, 500, 502, 503, 504) and attempt < retries:
                # anonymous pulls are limited (~100 GET/h); wait out the window rather than fail
                wait = float(e.headers.get("Retry-After") or 600) if e.code == 429 else 0
                sleep(min(max(wait, min(2 ** attempt * 2, 120)), 3700))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt >= retries:
                raise
            sleep(min(2 ** attempt * 2, 120))
    raise RuntimeError("unreachable")


class Registry:
    def __init__(self, fetch: Callable = http):
        self.fetch = fetch
        self.tokens: dict[str, str] = {}

    def token(self, repo: str, refresh: bool = False) -> str:
        if refresh or repo not in self.tokens:
            url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
            _, body = self.fetch("GET", url)
            self.tokens[repo] = json.loads(body)["token"]
        return self.tokens[repo]

    def manifest(self, repo: str, ref: str, method: str):
        url = f"{REGISTRY}/{repo}/manifests/{ref}"
        try:
            return self.fetch(method, url, {"Authorization": f"Bearer {self.token(repo)}", "Accept": ACCEPT})
        except urllib.error.HTTPError as e:
            if e.code != 401:
                raise
            return self.fetch(method, url, {"Authorization": f"Bearer {self.token(repo, True)}", "Accept": ACCEPT})

    def platform_of(self, image: str, index_digest: str) -> str:
        """GET the index by digest and return its linux/amd64 child digest."""
        headers, body = self.manifest(image.removesuffix(":latest"), index_digest, "GET")
        return parse_manifest_response({**headers, "docker-content-digest": index_digest}, body)["platform_digest"]

    def resolve(self, image: str, *, platform: bool = True) -> dict:
        """image like 'swebench/sweb...' (tag latest). Raises NotFound when the tag does not exist."""
        repo = image.removesuffix(":latest")
        headers, body = self.manifest(repo, "latest", "GET" if platform else "HEAD")
        return parse_manifest_response(headers, body)


def parse_manifest_response(headers: dict, body: bytes) -> dict:
    """Pure: turn a manifest response into {manifest_digest, manifest_kind, platform_digest?}."""
    digest = headers.get("docker-content-digest", "")
    if not DIGEST_RE.fullmatch(digest):
        raise ValueError(f"missing/invalid Docker-Content-Digest: {digest!r}")
    ctype = headers.get("content-type", "").split(";")[0].strip()
    if ctype in INDEX_TYPES:
        out = {"manifest_digest": digest, "manifest_kind": "index"}
        if body:
            platform = None
            for m in json.loads(body).get("manifests", []):
                p = m.get("platform") or {}
                if p.get("os") == "linux" and p.get("architecture") == "amd64":
                    platform = m.get("digest")
                    break
            if not platform or not DIGEST_RE.fullmatch(platform):
                raise ValueError("image index has no linux/amd64 manifest")
            out["platform_digest"] = platform
        return out
    if ctype in MANIFEST_TYPES:
        return {"manifest_digest": digest, "manifest_kind": "manifest"}
    raise ValueError(f"unexpected manifest media type {ctype!r}")


def all_instance_ids() -> list[str]:
    manifest = splits.load_manifest()
    return [row["instance_id"] for part in ("train", "val", "test") for row in manifest["partitions"][part]]


def load_v2(path: pathlib.Path, revision: str, v1: dict | None = None) -> dict:
    if path.exists():
        doc = json.loads(path.read_text())
        if doc.get("version") != 1 or doc.get("dataset_revision") != revision:
            raise ValueError("existing v2 manifest has wrong version/dataset revision")
        return doc
    return {"version": 1, "dataset_revision": revision, "images": {}}


def carry_over(entry: dict, v1_entry: dict | None) -> dict:
    """Keep a v1 entry's verification state when the digest matches."""
    if v1_entry and v1_entry.get("manifest_digest") == entry["manifest_digest"]:
        for k in ("verified", "sif_sha256", "verification"):
            if k in v1_entry:
                entry[k] = v1_entry[k]
    return entry


def write_atomic(path: pathlib.Path, doc: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    tmp.replace(path)


def pin_all(ids: list[str], doc: dict, v1: dict, registry: Registry, *, platform: bool = True,
            save: Callable[[dict], None] = lambda d: None, log=print) -> list[str]:
    """Pin every id missing from doc['images']; return the ids whose image does not exist on Docker Hub."""
    missing, done = [], 0
    for iid in ids:
        if iid in doc["images"]:
            continue
        image = splits.image_name(iid).removesuffix(":latest")
        try:
            info = registry.resolve(image, platform=platform)
        except NotFound:
            missing.append(iid)
            log(f"MISSING {iid}")
            continue
        entry = {"image": image, **info, "verified": False}
        doc["images"][iid] = carry_over(entry, v1.get(iid))
        done += 1
        if done % 20 == 0:
            save(doc)
            log(f"pinned {done} ({len(doc['images'])} total)")
    save(doc)
    return missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, default=V2)
    ap.add_argument("--v1", type=pathlib.Path, default=V1)
    ap.add_argument("--fill-platform", action="store_true",
                    help="after pinning, GET index manifests that lack platform_digest (slow: GETs are rate limited)")
    ap.add_argument("--no-platform", action="store_true", help="HEAD only: skip resolving the amd64 child digest")
    args = ap.parse_args(argv)
    manifest = splits.load_manifest()
    revision = manifest["revision"]
    v1 = json.loads(args.v1.read_text()).get("images", {}) if args.v1.exists() else {}
    ids = all_instance_ids()
    doc = load_v2(args.out, revision)
    missing = pin_all(ids, doc, v1, Registry(), platform=not args.no_platform,
                      save=lambda d: write_atomic(args.out, d))
    if args.fill_platform:
        reg = Registry()
        for iid, e in doc["images"].items():
            if e["manifest_kind"] == "index" and "platform_digest" not in e:
                e["platform_digest"] = reg.platform_of(e["image"], e["manifest_digest"])
                write_atomic(args.out, doc)
    kinds: dict[str, int] = {}
    for e in doc["images"].values():
        kinds[e["manifest_kind"]] = kinds.get(e["manifest_kind"], 0) + 1
    print(f"resolved={len(doc['images'])}/{len(ids)} missing={len(missing)} kinds={kinds}")
    for iid in missing:
        print("missing:", iid)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
