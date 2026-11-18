"""Require protected in-memory scratch for confidential provisioning deployments."""
from __future__ import annotations

import os
from pathlib import Path
import re
import stat


def _mount_type(path: Path, mountinfo: str):
    selected, depth = None, -1
    for line in mountinfo.splitlines():
        left, right = line.split(" - ", 1)
        escaped = left.split()[4]
        mount = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), escaped))
        if path.is_relative_to(mount) and len(mount.parts) > depth:
            selected, depth = right.split()[0], len(mount.parts)
    return selected


def require_confidential_scratch(root: Path):
    """An OS policy check, not hardware attestation of the memory itself.

    The accepted confidential image/VM must protect RAM. Requiring tmpfs with
    swap disabled avoids silently spilling decrypted provider artifacts to disk.
    """
    root = root.resolve(strict=True)
    info = root.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("confidential scratch must be an owned directory with mode 0700")
    try:
        fs = _mount_type(root, Path('/proc/self/mountinfo').read_text())
        swaps = Path('/proc/swaps').read_text().splitlines()
    except (OSError, ValueError) as error:
        raise ValueError("cannot establish confidential scratch policy") from error
    if fs != 'tmpfs' or len([line for line in swaps[1:] if line.strip()]) > 0:
        raise ValueError("confidential scratch requires tmpfs and disabled swap")


def require_capacity(root: Path, padded_bytes: int):
    """Conservative preflight, not a guarantee against concurrent space usage.

    The cap already bounds the whole deduplicated bundle. Padding is not stored,
    but reserving the cap plus metadata/headroom avoids optimistic space claims.
    """
    stat = os.statvfs(root)
    available = stat.f_bavail * stat.f_frsize
    required = padded_bytes + max(16 * 1024**2, padded_bytes // 20)
    if available < required:
        raise ValueError("insufficient private scratch capacity for the public envelope")

