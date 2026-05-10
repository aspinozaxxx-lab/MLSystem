from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksum(path: str | Path, checksum_path: str | Path | None = None) -> str:
    checksum = sha256_file(path)
    target = Path(checksum_path) if checksum_path is not None else Path(path).with_suffix(Path(path).suffix + ".sha256")
    target.write_text(checksum + "\n", encoding="utf-8")
    return checksum


def checksum_matches(path: str | Path, checksum_path: str | Path) -> bool:
    path_obj = Path(path)
    checksum_obj = Path(checksum_path)
    if not path_obj.exists() or not checksum_obj.exists():
        return False
    expected = checksum_obj.read_text(encoding="utf-8").strip()
    return bool(expected) and sha256_file(path_obj) == expected
