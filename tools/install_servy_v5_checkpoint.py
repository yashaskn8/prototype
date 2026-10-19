"""Install an already-approved Servy V5 CrossEncoder checkpoint without Git or network access.

The installer verifies the fixed V5 weight identity, refuses symlinks and
existing targets, and stages a complete local copy before publishing it.
It does not enable production serving or attest independent accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

V5_WEIGHTS_SHA256 = "b1cf23ba0806f0057d73a5ab93fde88d634fc278f71b27f3a2a915afd8f8bcda"
DEFAULT_TARGET = Path(__file__).resolve().parent.parent / "ml_models" / "checkpoints" / "servy_v5" / "best_model"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source(source: Path, expected_sha256: str = V5_WEIGHTS_SHA256) -> None:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Source must be a real directory, not a symlink")
    for root, directories, files in os.walk(source, followlinks=False):
        for name in directories + files:
            if (Path(root) / name).is_symlink():
                raise ValueError("Checkpoint contains symlinks; refusing installation")
    for filename in ("model.safetensors", "config.json"):
        if not (source / filename).is_file():
            raise ValueError(f"V5 checkpoint missing {filename}")
    if file_sha256(source / "model.safetensors") != expected_sha256:
        raise ValueError("Checkpoint weights do not match the approved V5 SHA-256")


def install_checkpoint(source: Path, target: Path = DEFAULT_TARGET,
                       expected_sha256: str = V5_WEIGHTS_SHA256) -> Path:
    source = Path(source).expanduser().absolute()
    target = Path(target).expanduser().absolute()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Source and target directories must not overlap")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {target}")
    validate_source(source, expected_sha256)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v5-install-", dir=target.parent) as stage:
        staged = Path(stage) / "best_model"
        shutil.copytree(source, staged, symlinks=False)
        validate_source(staged, expected_sha256)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Checkpoint destination appeared during installation: {target}")
        staged.rename(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Local V5 best_model directory")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    location = install_checkpoint(args.source, args.target)
    print(f"SERVY V5 CHECKPOINT INSTALLED AND SHA256 VERIFIED: {location}")
    print("Activation requires explicit local configuration and a successful Django inference smoke check.")


if __name__ == "__main__":
    main()
