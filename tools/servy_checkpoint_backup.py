"""Save a Servy checkpoint to user-approved durable storage and verify all bytes.

Usage (Colab, after mounting APPROVED persistent storage):
  python tools/servy_checkpoint_backup.py \
    --source /content/servy_reranker_v4 \
    --destination /content/drive/MyDrive/APPROVED/servy_reranker_v4

No model training, Git writes or cloud API usage. Never overwrites existing backup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
import zipfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as inp:
        for data in iter(lambda: inp.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def collect(path):
    """Deterministic content inventory, rejecting symbolic links."""
    if path.is_symlink():
        raise ValueError("Source contains a symbolic link")
    if path.is_file():
        return [(path.name, path)]
    if not path.is_dir():
        raise FileNotFoundError(path)
    items = []
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"Source contains symlink: {item}")
        if item.is_file():
            items.append((item.relative_to(path).as_posix(), item))
    if not items:
        raise ValueError("Cannot back up an empty checkpoint directory")
    return items


def is_temporary_colab_path(path):
    normalized = str(path).replace("\\", "/").rstrip("/")
    return normalized == "/content" or (
        normalized.startswith("/content/")
        and not normalized.startswith("/content/drive/")
    )


def backup_checkpoint(source, destination):
    src, dst = Path(source), Path(destination)
    if not dst.is_absolute():
        raise ValueError("Backup destination must be an absolute approved persistent path")
    # In Colab, /content/drive can exist as an ordinary empty directory after
    # a disconnected runtime. It is NOT persistent until the Drive mount is live.
    target_path = dst.as_posix()
    if (target_path == "/content/drive" or target_path.startswith("/content/drive/")):
        if not os.path.ismount("/content/drive"):
            raise ValueError(
                "Google Drive is not mounted at /content/drive; refusing ephemeral backup"
            )
    if not src.exists():
        raise FileNotFoundError(f"Source checkpoint missing: {src}")
    if is_temporary_colab_path(dst):
        raise ValueError("Backup destination is ephemeral /content; use approved durable storage")
    if dst.exists():
        raise FileExistsError(f"Backup already exists (refusing overwrite): {dst}")
    if src.is_dir() and src.resolve() in dst.resolve().parents:
        raise ValueError("Destination must not be nested inside the source checkpoint")
    if not dst.parent.is_dir():
        raise FileNotFoundError(f"Backup destination parent does not exist: {dst.parent}")
    entries = collect(src)
    stage = dst.with_name(dst.name + ".partial-" + uuid.uuid4().hex[:8])
    if stage.exists():
        raise FileExistsError(stage)
    try:
        if src.is_file():
            shutil.copy2(src, stage)
        else:
            shutil.copytree(src, stage, symlinks=False)
        staged = ({src.name: stage} if src.is_file() else dict(collect(stage)))
        expected = {name: (path.stat().st_size, sha256(path)) for name, path in entries}
        actual = {name: (path.stat().st_size, sha256(path)) for name, path in staged.items()}
        if expected != actual:
            raise IOError("Checkpoint backup verification failed (size or hash mismatch)")
        if src.is_file() and zipfile.is_zipfile(src):
            with zipfile.ZipFile(stage) as archive:
                corrupt = archive.testzip()
                if corrupt is not None:
                    raise IOError(f"Invalid ZIP entry: {corrupt}")
        os.replace(stage, dst)
        rechecked = ({src.name: dst} if src.is_file() else dict(collect(dst)))
        final = {name: (path.stat().st_size, sha256(path)) for name, path in rechecked.items()}
        if expected != final:
            raise IOError("Post-rename verification failed")
        return {
            "status": "VERIFIED_BACKUP",
            "source": str(src), "destination": str(dst),
            "files": len(expected),
            "bytes": sum(size for size, _ in expected.values()),
            "hashes": {name: digest for name, (_, digest) in expected.items()},
        }
    finally:
        if stage.is_dir():
            shutil.rmtree(stage)
        elif stage.is_file():
            stage.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    result = backup_checkpoint(args.source, args.destination)
    print(json.dumps(result, indent=2))
    print("Backup verified. Only now may you delete the Colab runtime.")


if __name__ == "__main__":
    main()
