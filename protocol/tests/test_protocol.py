from copy import deepcopy
import io
import json
from pathlib import Path
import struct

import pytest

from ct_core.interfaces import ArtifactPackage
from util.dynamic_test_environment import StaticEnvironment

from protocol.attestation import local_test_pair
from protocol.bundle import BundlePlan
from protocol.codec import CHUNK, canonical, digest
from protocol import stream
from protocol.demo import ArithmeticAgentDriver, NegativeTapDriver, environment, packages
from protocol.protocol import Provider, Workload, make_spec, verify_certificate
from protocol.errors import ABORT, ProtocolError


@pytest.fixture
def session(tmp_path):
    ap, tp = packages(tmp_path / "provider")
    plan = BundlePlan(ap, tp, agent_drivers=(ArithmeticAgentDriver,), tap_drivers=(NegativeTapDriver,))
    spec, padded = make_spec(environment(), component_bytes=64*1024)
    signer, verifier = local_test_pair()
    workload = Workload(signer, sid="test-session", agent_drivers=(ArithmeticAgentDriver,),
                        tap_drivers=(NegativeTapDriver,), environment_drivers=(StaticEnvironment,), scratch_root=tmp_path,
                        storage_mode="local-test")
    return dict(plan=plan, ap=ap, tp=tp, spec=spec, padded=padded, signer=signer, verifier=verifier,
                workload=workload, root=tmp_path)


def exchange(s):
    proposal = s['workload'].install_environment(s['spec'], s['padded'])
    provider = Provider(s['plan'], s['verifier'], expected_spec=s['spec'], sid="test-session")
    ciphertext = io.BytesIO()
    provider.write_submission(proposal, ciphertext)
    return proposal, provider, ciphertext.getvalue()


def another_workload(s):
    return Workload(s['signer'], sid="test-session", agent_drivers=(ArithmeticAgentDriver,),
                    tap_drivers=(NegativeTapDriver,), environment_drivers=(StaticEnvironment,), scratch_root=s['root'],
                    storage_mode="local-test")


def test_complete_private_exchange_and_public_verification(session):
    s = session
    proposal, provider, encrypted = exchange(s)
    assert "CERTIFIER_PRIVATE_CASE_b3e484dc" not in canonical(proposal).decode()
    assert b"PROVIDER_PRIVATE_SETTINGS_7b181846" not in encrypted
    cert = s['workload'].certify(io.BytesIO(encrypted))
    assert provider.accept(cert)
    assert verify_certificate(cert, s['verifier'], expected_spec=s['spec'], expected_id=s['plan'].identity)
    assert cert['m'] == {'fn': 0., 'fp': 0.}
    assert cert['id'] == s['plan'].identity
    assert all(marker not in canonical(cert).decode() for marker in
               ['CERTIFIER_PRIVATE_CASE', 'PROVIDER_PRIVATE_SETTINGS', 'private_negative', 'threshold.bin'])
    assert s['workload']._secret_key is s['workload']._environment is None
    assert not list(s['root'].glob('ct-private-*'))
    assert s['workload'].state == 'complete'


def test_no_submission_before_tests_and_one_result_only(session):
    s = session
    with pytest.raises(ValueError):
        s['workload'].certify(io.BytesIO())
    _, provider, encrypted = exchange(s)
    cert = s['workload'].certify(io.BytesIO(encrypted))
    assert provider.accept(cert)
    with pytest.raises(ValueError):
        s['workload'].certify(io.BytesIO(encrypted))
    with pytest.raises(ValueError):
        s['workload'].install_environment(s['spec'], s['padded'])


def test_stable_identity_across_fresh_sessions_and_ciphertexts(session):
    s = session
    p1, provider1, c1 = exchange(s)
    cert1 = s['workload'].certify(io.BytesIO(c1))
    s2 = dict(s, workload=another_workload(s))
    p2, provider2, c2 = exchange(s2)
    cert2 = s2['workload'].certify(io.BytesIO(c2))
    assert c1 != c2 and p1['eid'] != p2['eid'] and p1['pk'] != p2['pk']
    assert cert1['id'] == cert2['id']
    assert provider1.accept(cert1) and provider2.accept(cert2)
    assert not provider1.accept(cert2)


def test_wrong_session_replay_fails_before_loader(session):
    s = session
    _, _, encrypted = exchange(s)
    other = another_workload(s)
    other.install_environment(s['spec'], s['padded'])
    ArithmeticAgentDriver.loads = 0
    with pytest.raises(Exception):
        other.certify(io.BytesIO(encrypted))
    assert ArithmeticAgentDriver.loads == 0 and other.state == 'aborted'
    assert not list(s['root'].glob('ct-private-*'))


@pytest.mark.parametrize('alter', ['signature', 'key', 'spec', 'sid', 'extra'])
def test_provider_rejects_wrong_attested_proposal(session, alter):
    s = session
    p = s['workload'].install_environment(s['spec'], s['padded'])
    if alter == 'signature': p['sig'] = '00'*64
    elif alter == 'key': p['pk'] = '00'*32
    elif alter == 'spec': p['spec']['component_bytes'] += 1
    elif alter == 'sid': p['sid'] = 'other'
    else: p['unexpected'] = True
    target = io.BytesIO()
    provider = Provider(s['plan'], s['verifier'], expected_spec=s['spec'], sid="test-session")
    with pytest.raises(ValueError):
        provider.write_submission(p, target)
    assert target.getvalue() == b''


@pytest.mark.parametrize('mutation', ['middle', 'last', 'truncated', 'trailing', 'prefix'])
def test_tamper_and_truncation_never_load_partial_artifacts(session, mutation):
    s = session
    _, _, encrypted = exchange(s)
    data = bytearray(encrypted)
    if mutation == 'middle': data[len(data)//2] ^= 1
    elif mutation == 'last': data[-1] ^= 1
    elif mutation == 'truncated': del data[-20:]
    elif mutation == 'trailing': data += b'junk'
    else: data[0] ^= 1
    ArithmeticAgentDriver.loads = 0
    with pytest.raises(Exception):
        s['workload'].certify(io.BytesIO(data))
    assert ArithmeticAgentDriver.loads == 0
    assert s['workload'].state == 'aborted'
    assert s['workload']._secret_key is s['workload']._environment is None
    assert not list(s['root'].glob('ct-private-*'))


def test_changed_source_file_does_not_certify_old_manifest(session):
    s = session
    source = Path(dict(s['ap'].artifacts)['model/settings.json'])
    data = json.loads(source.read_text())
    data['multiplier'] = 7
    source.write_bytes(canonical(data))
    changed = BundlePlan(s['ap'], s['tp'], agent_drivers=(ArithmeticAgentDriver,), tap_drivers=(NegativeTapDriver,))
    assert changed.identity != s['plan'].identity
    _, _, encrypted = exchange(s)
    ArithmeticAgentDriver.loads = 0
    with pytest.raises(ProtocolError, match=ABORT):
        s['workload'].certify(io.BytesIO(encrypted))
    assert ArithmeticAgentDriver.loads == 0


def test_tap_change_changes_stable_identity(session):
    s = session
    changed = ArtifactPackage(NegativeTapDriver.DRIVER, {}, (('threshold.bin', struct.pack('!d', 2.)),))
    plan = BundlePlan(s['ap'], changed, agent_drivers=(ArithmeticAgentDriver,), tap_drivers=(NegativeTapDriver,))
    assert plan.identity != s['plan'].identity


def forged_submission(s, mutate, *, bad_padding=False):
    proposal = s['workload'].install_environment(s['spec'], s['padded'])
    meta = json.loads(s['plan']._encoded)
    mutate(meta)
    encoded = canonical(meta)
    # The real plan's small files, in the same order, with independently forged inert metadata.
    body = Path(dict(s['ap'].artifacts)['model/settings.json']).read_bytes() + dict(s['tp'].artifacts)['threshold.bin']
    payload = len(encoded).to_bytes(8, 'big') + encoded + body
    payload += bytes(s['spec']['component_bytes']-len(payload))
    if bad_padding:
        payload = payload[:-1] + b'\x01'
    target = io.BytesIO()
    stream.encrypt(bytes.fromhex(proposal['pk']), bytes.fromhex(digest(canonical(proposal))),
                   [payload], len(payload), target)
    return target.getvalue()


@pytest.mark.parametrize('attack', ['path', 'driver', 'code', 'size', 'padding'])
def test_authenticated_but_invalid_provider_manifest_rejected(session, attack):
    s = session
    def mutate(meta):
        record = meta['components']['agent']
        if attack == 'path':
            record['artifacts']['../escape'] = record['artifacts'].pop('model/settings.json')
            meta['storage']['agent']['../escape'] = meta['storage']['agent'].pop('model/settings.json')
        elif attack == 'driver': record['driver'] = 'unregistered-python'
        elif attack == 'code': record['code'] = {}
        elif attack == 'size': record['artifacts']['model/settings.json']['bytes'] = 2**40
    encrypted = forged_submission(s, mutate, bad_padding=attack == 'padding')
    ArithmeticAgentDriver.loads = 0
    with pytest.raises(ValueError):
        s['workload'].certify(io.BytesIO(encrypted))
    assert ArithmeticAgentDriver.loads == 0
    assert not (s['root'] / 'escape').exists()


def test_execution_failure_releases_no_partial_certificate(session):
    s = session
    ap, tp = packages(s['root'] / 'failing-provider', fail_at=-2)
    s['plan'] = BundlePlan(ap, tp, agent_drivers=(ArithmeticAgentDriver,), tap_drivers=(NegativeTapDriver,))
    _, provider, encrypted = exchange(s)
    with pytest.raises(ProtocolError, match=ABORT):
        s['workload'].certify(io.BytesIO(encrypted))
    assert s['workload'].state == 'aborted' and not provider.accept({})
    assert not list(s['root'].glob('ct-private-*'))


@pytest.mark.parametrize('field', ['id', 'eid', 'm_ref', 'sig', 'fn', 'extra'])
def test_certificate_tampering_and_wrong_identity_rejected(session, field):
    s = session
    _, provider, encrypted = exchange(s)
    certificate = s['workload'].certify(io.BytesIO(encrypted))
    cert = deepcopy(certificate)
    if field in ('id', 'eid', 'm_ref', 'sig'): cert[field] = '0'*len(cert[field])
    elif field == 'fn': cert['m']['fn'] = .25
    else: cert['extra'] = 'private output'
    assert not provider.accept(cert)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, 2, True])
def test_invalid_loss_values_are_never_accepted(session, value):
    s = session
    _, provider, encrypted = exchange(s)
    cert = s['workload'].certify(io.BytesIO(encrypted))
    cert['m']['fn'] = value
    assert not provider.accept(cert)


def test_public_verifier_does_not_accept_a_self_chosen_anchor(session):
    s = session
    _, _, encrypted = exchange(s)
    cert = s['workload'].certify(io.BytesIO(encrypted))
    _, wrong = local_test_pair()
    assert not verify_certificate(cert, wrong, expected_spec=s['spec'], expected_id=s['plan'].identity)


def test_multichunk_transfer_on_disk_not_a_model_load(session):
    s = session
    payload = s['root'] / 'payload.bin'
    with payload.open('wb') as out:
        for _ in range(32): out.write(b'x'*CHUNK)
    ap = ArtifactPackage(s['ap'].driver, {}, (*s['ap'].artifacts, ('model/payload.bin', str(payload))))
    s['plan'] = BundlePlan(ap, s['tp'], agent_drivers=(ArithmeticAgentDriver,), tap_drivers=(NegativeTapDriver,))
    s['spec'], s['padded'] = make_spec(environment(), component_bytes=40*CHUNK)
    proposal = s['workload'].install_environment(s['spec'], s['padded'])
    provider = Provider(s['plan'], s['verifier'], expected_spec=s['spec'], sid='test-session')
    path = s['root'] / 'cipher.enc'
    with path.open('wb') as out: provider.write_submission(proposal, out)
    with path.open('rb') as inp: cert = s['workload'].certify(inp)
    assert provider.accept(cert)
    assert not list(s['root'].glob('ct-private-*'))

