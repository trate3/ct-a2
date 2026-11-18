import base64
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import secrets
import time

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from nacl.public import PrivateKey
from nacl.signing import SigningKey
import pytest

from ct_core.protocol.attestor import GcpAttestationVerifier
from protocol.attestation import accepted_gcp_verifier
from protocol.codec import CHUNK, canonical, decode
from protocol import stream
from protocol.storage import _mount_type, require_confidential_scratch


@pytest.mark.parametrize('mode', ['reorder', 'duplicate', 'drop', 'wrong_context', 'wrong_length'])
def test_stream_rejects_sequence_and_scope_changes(mode):
    secret = PrivateKey.generate()
    context = secrets.token_bytes(32)
    target = io.BytesIO()
    length = 3*CHUNK
    stream.encrypt(bytes(secret.public_key), context, [b'a'*CHUNK, b'b'*CHUNK, b'c'*CHUNK], length, target)
    encoded = target.getvalue()
    head = len(stream.MAGIC)+stream.SEALED_BYTES+stream.HEADER_BYTES
    frame = 4+CHUNK+stream.OVERHEAD
    first, second, third = [encoded[head+i*frame:head+(i+1)*frame] for i in range(3)]
    body = encoded
    if mode == 'reorder': body = encoded[:head]+second+first+third
    elif mode == 'duplicate': body = encoded[:head]+first+first+third
    elif mode == 'drop': body = encoded[:head]+first+third
    elif mode == 'wrong_context': context = secrets.token_bytes(32)
    else: length += 1
    with pytest.raises(Exception):
        list(stream.decrypt(secret, context, length, io.BytesIO(body)))


def test_stream_length_hides_private_boundaries_and_encryption_is_fresh():
    sk = PrivateKey.generate()
    context = secrets.token_bytes(32)
    outputs = []
    for marker in (b'short', b'a'*3000):
        out = io.BytesIO()
        stream.encrypt(bytes(sk.public_key), context, [marker, bytes(CHUNK-len(marker))], CHUNK, out)
        outputs.append(out.getvalue())
    assert len(outputs[0]) == len(outputs[1]) and outputs[0] != outputs[1]


def test_transport_never_requires_an_unbounded_read():
    class Bounded(io.BytesIO):
        def read(self, n=-1):
            assert 0 <= n <= CHUNK+stream.OVERHEAD
            return super().read(n)
    sk, context = PrivateKey.generate(), secrets.token_bytes(32)
    out = io.BytesIO()
    stream.encrypt(bytes(sk.public_key), context, [bytes(CHUNK)]*3, CHUNK*3, out)
    assert sum(len(b) for b in stream.decrypt(sk, context, 3*CHUNK, Bounded(out.getvalue()))) == 3*CHUNK


def test_strict_json_rejects_duplicate_keys_and_nonfinite_literals():
    for raw in [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}']:
        with pytest.raises(ValueError): decode(raw)


def test_mount_policy_chooses_actual_containing_mount():
    mounts = '1 0 0:1 / / rw - ext4 disk rw\n2 1 0:2 / /private rw - tmpfs tmpfs rw\n3 1 0:3 / /private/disk rw - ext4 disk rw\n'
    assert _mount_type(Path('/private/work'), mounts) == 'tmpfs'
    assert _mount_type(Path('/private/disk/work'), mounts) == 'ext4'
    assert _mount_type(Path('/private-sibling'), mounts) == 'ext4'


def test_confidential_scratch_rejects_ordinary_disk(tmp_path):
    # Simulate the mount record, without changing mounts, swap, or system state.
    from unittest.mock import patch
    original = Path.read_text
    def read(path, *a, **kw):
        if str(path) == '/proc/self/mountinfo': return '1 0 0:1 / / rw - ext4 disk rw\n'
        if str(path) == '/proc/swaps': return 'Filename Type Size Used Priority\n'
        return original(path, *a, **kw)
    with patch.object(Path, 'read_text', read), pytest.raises(ValueError, match='tmpfs'):
        require_confidential_scratch(tmp_path)


@pytest.fixture(scope='module')
def attestation_key():
    # Small local verifier test, not an attestation-service request.
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.mark.parametrize('change', ['none', 'image', 'audience', 'debug', 'software', 'nonce', 'signature'])
def test_gcp_policy_verifies_signature_nonce_and_explicit_production_policy(attestation_key, change):
    image, audience = 'sha256:'+'a'*64, 'test-audience'
    key = bytes(SigningKey.generate().verify_key)
    claims = {'iss': GcpAttestationVerifier.ISSUER, 'aud': audience, 'exp': int(time.time())+600,
              'eat_nonce': [base64.b64encode(hashlib.sha256(key).digest()).decode()],
              'dbgstat': 'disabled-since-boot', 'swname': 'CONFIDENTIAL_SPACE',
              'submods': {'container': {'image_digest': image}}}
    if change == 'image': claims['submods']['container']['image_digest'] = 'sha256:'+'b'*64
    elif change == 'audience': claims['aud'] = 'wrong'
    elif change == 'debug': claims['dbgstat'] = 'enabled'
    elif change == 'software': claims['swname'] = 'GCE'
    elif change == 'nonce': claims['eat_nonce'] = []
    signing_key = rsa.generate_private_key(public_exponent=65537, key_size=2048) if change == 'signature' else attestation_key
    token = jwt.encode(claims, signing_key, algorithm='RS256')
    class Jwks:
        def get_signing_key_from_jwt(self, token):
            class Key: key = attestation_key.public_key()
            return Key()
    kwargs = dict(accepted_image=image, audience=audience, jwks_client=Jwks())
    if change == 'none':
        verifier = accepted_gcp_verifier(token, key, **kwargs)
        assert verifier.measurement() == image
    else:
        with pytest.raises(Exception): accepted_gcp_verifier(token, key, **kwargs)

