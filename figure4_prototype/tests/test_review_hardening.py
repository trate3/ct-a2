from collections import namedtuple
import io
import json
import os
from pathlib import Path
import socket
import time

import pytest

from ct_core import certification as core
from ct_core.interfaces import ArtifactPackage, TapDriver
from figure4_prototype.attestation import LocalSigner, local_test_pair
from figure4_prototype.bundle import BundlePlan
from figure4_prototype.codec import CHUNK, ChunkReader, canonical, digest
from figure4_prototype.demo import ArithmeticAgentDriver, NegativeTapDriver, NegativeTap, environment, packages
from figure4_prototype.errors import ABORT, ProtocolError
from figure4_prototype.protocol import Provider, Workload, make_spec
from figure4_prototype.storage import require_capacity
from figure4_prototype import stream


class SharedFileTapDriver(TapDriver):
    DRIVER = 'figure4-test-shared-file/v1'
    paths = []

    @classmethod
    def from_package(cls, config, artifacts):
        cls.paths.append(artifacts['model/settings.json'])
        obj = cls()
        return obj

    def new_tap(self, randomness):
        return NegativeTap(.5)


def setup(tmp_path, *, shared=False):
    ap, tp = packages(tmp_path/'provider')
    if shared:
        tp = ArtifactPackage(SharedFileTapDriver.DRIVER, {}, ap.artifacts)
    tap_cls = SharedFileTapDriver if shared else NegativeTapDriver
    plan = BundlePlan(ap, tp, agent_drivers=(ArithmeticAgentDriver,), tap_drivers=(tap_cls,))
    signer, verifier = local_test_pair()
    spec, padded = make_spec(environment(), component_bytes=64*1024)
    workload = Workload(signer, sid='hardening-test', agent_drivers=(ArithmeticAgentDriver,), tap_drivers=(tap_cls,),
                        environment_drivers=(type(environment()),), scratch_root=tmp_path, storage_mode='local-test')
    return ap, tp, plan, signer, verifier, spec, padded, workload


def test_shared_content_sent_once_identity_unchanged_and_loader_paths_shared(tmp_path, monkeypatch):
    ap, tp, plan, signer, verifier, spec, padded, workload = setup(tmp_path, shared=True)
    size = Path(ap.artifacts[0][1]).stat().st_size
    assert plan.unique_artifact_bytes == size
    meta = json.loads(plan._encoded)
    assert plan.minimum_bytes == 8+len(plan._encoded)+size
    assert plan.identity == digest(canonical(meta))
    loaded = []
    original = ArithmeticAgentDriver.from_package.__func__
    def spy(cls, config, artifacts):
        loaded.append(artifacts['model/settings.json'])
        return original(cls, config, artifacts)
    monkeypatch.setattr(ArithmeticAgentDriver, 'from_package', classmethod(spy))
    proposal = workload.install_environment(spec, padded)
    provider = Provider(plan, verifier, expected_spec=spec, sid='hardening-test')
    out = io.BytesIO()
    provider.write_submission(proposal, out)
    certificate = workload.certify(io.BytesIO(out.getvalue()))
    assert provider.accept(certificate)
    assert loaded[-1] == SharedFileTapDriver.paths[-1]
    assert not list(tmp_path.glob('figure4-private-*'))


def test_deduplication_is_by_content_not_source_path(tmp_path):
    ap, tp, plan, *_ = setup(tmp_path, shared=True)
    separate = tmp_path/'a-separate-source.json'
    separate.write_bytes(Path(ap.artifacts[0][1]).read_bytes())
    tp = ArtifactPackage(tp.driver, {}, (('model/settings.json',str(separate)),))
    other = BundlePlan(ap, tp, agent_drivers=(ArithmeticAgentDriver,), tap_drivers=(SharedFileTapDriver,))
    assert plan.identity == other.identity and plan.unique_artifact_bytes == other.unique_artifact_bytes


def test_tampered_inline_probe_rejected_before_any_loader(tmp_path):
    ap, tp, plan, signer, verifier, spec, padded, workload = setup(tmp_path)
    proposal = workload.install_environment(spec, padded)
    # Valid encryption/authentication, inconsistent inline probe bytes vs manifest.
    pieces = list(plan.plaintext(spec['component_bytes']))
    payload = bytearray(b''.join(pieces))
    first_inline = 8+len(plan._encoded)+Path(ap.artifacts[0][1]).stat().st_size
    payload[first_inline] ^= 1
    out = io.BytesIO()
    stream.encrypt(bytes.fromhex(proposal['pk']), bytes.fromhex(digest(canonical(proposal))),
                   [payload], len(payload), out)
    ArithmeticAgentDriver.loads = 0
    with pytest.raises(ProtocolError, match=ABORT): workload.certify(io.BytesIO(out.getvalue()))
    assert ArithmeticAgentDriver.loads == 0


def test_capacity_fails_before_proposal_key_generation(tmp_path, monkeypatch):
    *_, spec, padded, workload = setup(tmp_path)
    monkeypatch.setattr('figure4_prototype.protocol.require_capacity', lambda *a: (_ for _ in ()).throw(ValueError('no space')))
    with pytest.raises(ProtocolError, match=ABORT): workload.install_environment(spec, padded)
    assert workload._secret_key is None and workload.state=='aborted'


def test_capacity_counts_whole_envelope_once_plus_headroom(tmp_path, monkeypatch):
    Stat = namedtuple('Stat','f_bavail f_frsize')
    monkeypatch.setattr(os, 'statvfs', lambda _:Stat(100,1024*1024))
    require_capacity(tmp_path, 80*1024*1024)
    with pytest.raises(ValueError): require_capacity(tmp_path, 90*1024*1024)


def test_confidential_mode_refuses_second_workload_in_same_process(tmp_path, monkeypatch):
    import figure4_prototype.protocol as protocol
    monkeypatch.setattr(protocol, 'require_confidential_scratch', lambda _:None)
    monkeypatch.setattr(Workload, '_confidential_session_claimed', False)
    local, _ = local_test_pair()
    class ExternalSigner:
        measurement = local.measurement
        sign = local.sign
    kwargs = dict(sid='confidential-process-test', agent_drivers=(), tap_drivers=(), environment_drivers=(),
                  scratch_root=tmp_path, storage_mode='confidential-tmpfs')
    Workload(ExternalSigner(), **kwargs)
    with pytest.raises(ValueError, match='fresh process'): Workload(ExternalSigner(), **kwargs)


@pytest.mark.parametrize('prefix', [b'', stream.MAGIC[:3]])
def test_stalled_peer_socket_obeys_deadline_and_never_loads(tmp_path, prefix):
    *_, spec, padded, workload = setup(tmp_path)
    workload.install_environment(spec, padded)
    reader, writer = socket.socketpair()
    try:
        if prefix: writer.sendall(prefix)
        ArithmeticAgentDriver.loads = 0
        started = time.monotonic()
        with pytest.raises(ProtocolError, match=ABORT):
            workload.certify(reader, deadline=started+.02)
        assert time.monotonic()-started < 1
        assert ArithmeticAgentDriver.loads==0 and workload.state=='aborted'
        assert workload._secret_key is workload._environment is None
    finally:
        reader.close()
        writer.close()


def test_generic_abort_does_not_expose_crypto_or_private_validation_error(tmp_path):
    *_, spec, padded, workload = setup(tmp_path)
    workload.install_environment(spec, padded)
    with pytest.raises(ProtocolError) as caught: workload.certify(io.BytesIO(b'wrong'))
    assert type(caught.value) is ProtocolError and str(caught.value)==ABORT
    assert caught.value.__suppress_context__


def test_public_adapter_returns_only_generic_abort(tmp_path):
    *_, spec, padded, workload = setup(tmp_path)
    workload.install_environment(spec, padded)
    assert workload.handle_submission(io.BytesIO(b'bad')) == {'type':'abort','version':'ct-figure4/2'}


def test_timing_stages_are_nested_not_summed_as_exclusive(tmp_path):
    ap,tp,plan,signer,verifier,spec,padded,workload = setup(tmp_path)
    proposal=workload.install_environment(spec,padded)
    provider=Provider(plan,verifier,expected_spec=spec,sid='hardening-test')
    out=io.BytesIO();provider.write_submission(proposal,out)
    from ct_core.protocol.telemetry import StageRecorder
    recorder=StageRecorder('stage-test')
    cert=workload.certify(io.BytesIO(out.getvalue()),stage_recorder=recorder)
    names={s.stage for s in recorder.records}
    assert {'provider_provisioning_total','stream_decrypt_write','artifact_verify','private_snapshot_layout','component_prepare_total'} <= names
    import inspect
    if 'stage_recorder' in inspect.signature(core.prepare_components).parameters:
        assert {'artifact_hash_pre','component_load','artifact_hash_post'} <= names
    assert provider.accept(cert)

