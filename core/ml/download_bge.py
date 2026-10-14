"""Explicit, verifiable download of BAAI/bge-small-en-v1.5 (public artifacts only)."""
import json, os, sys, time
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
    os.environ.pop(k, None)
from huggingface_hub import HfApi, snapshot_download

REPO = "BAAI/bge-small-en-v1.5"
info = HfApi().model_info(REPO, token=False)
sha = info.sha
lic = (info.card_data or {}).get("license") if info.card_data else None
files = sorted(s.rfilename for s in info.siblings)
print("repo", REPO, "sha", sha, "license", lic)
allow = ["*.json", "*.txt", "model.safetensors", "1_Pooling/*", "modules.json", "README.md"]
t = time.time()
path = snapshot_download(REPO, revision=sha, allow_patterns=allow, token=False)
print("downloaded_to", path, "secs", round(time.time() - t, 1))
print("local_files", sorted(os.listdir(path)))

from sentence_transformers import SentenceTransformer
m = SentenceTransformer(REPO, revision=sha, device="cpu", local_files_only=True)
e = m.encode(["Milk analyzer does not draw a sample.", "Gate interlock fault on lift."], normalize_embeddings=True)
import numpy as np
print("dim", m.get_sentence_embedding_dimension(), "max_seq_length", m.max_seq_length)
print("shape", e.shape, "finite", bool(np.isfinite(e).all()), "norms", np.round(np.linalg.norm(e, axis=1), 4).tolist())
print("pair_cosine", round(float(e[0] @ e[1]), 4))
json.dump({"repo": REPO, "revision": sha, "license": lic, "path": path,
           "dim": m.get_sentence_embedding_dimension(), "max_seq_length": m.max_seq_length},
          open(sys.argv[1], "w"), indent=2)
