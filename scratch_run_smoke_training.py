"""Isolated Synthetic BGE-Small Fine-Tuning Smoke Test (Branch B).

Under Phase 11 Branch B:
- No verified human labels exist (8 candidate queries are pending human review in candidates_for_review.jsonl).
- We execute a strictly isolated synthetic smoke test on BAAI/bge-small-en-v1.5 to verify the model loading path,
  optimizer steps, checkpoint save, and checkpoint reload.
- The resulting checkpoint is registered with promotion_eligible=False, permanently marked ineligible for production.
"""
import os
import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
django.setup()

from core.ml.training import finetune
from core.ml.embeddings import register_finetuned, get_profile, load_model

def main():
    print("=== STARTING PHASE 11 BRANCH B: SYNTHETIC FINE-TUNING SMOKE TEST ===")

    synthetic_train = [
        {
            "query_text": "Sample pump motor humming but no liquid drawn into chamber",
            "positive_text": "Milk Analyzer Fault Code E-102: Sample intake pump stall. Inspect peristaltic tubing for pinch blockage and clean the intake valve assembly before re-initiating auto-calibration.",
            "negative_text": "Safety interlock gate sensor bypass procedure for routine conveyor maintenance on model AeroLift-5000.",
        },
        {
            "query_text": "Optical sensor drift during fat percentage measurement",
            "positive_text": "Optical bench zero-drift calibration guide: Clean the quartz flow cell using approved deionized rinse solution. Re-zero wavelength spectrometer at 450nm and 680nm.",
            "negative_text": "Hydraulic fluid filter replacement instructions for high-pressure industrial lift pumps.",
        },
        {
            "query_text": "Centrifuge excessive vibration error code E-301",
            "positive_text": "Error E-301 indicates dynamic rotor imbalance. Verify symmetrical tube loading and inspect elastomeric vibration isolator dampers for elastomer wear or degradation.",
            "negative_text": "Thermostat control board fuse rating specifications for refrigerated dairy transport tankers.",
        },
        {
            "query_text": "Thermal block temperature controller communication timeout",
            "positive_text": "RS-485 serial communication fault between master controller and thermal block. Verify baud rate 9600-8-N-1 and inspect termination resistor continuity on pin 3.",
            "negative_text": "Air compressor regulator valve pressure adjustment procedure for pneumatic packaging lines.",
        },
    ]

    synthetic_val = [
        {
            "query_text": "Flow cell clogged after running unhomogenized whole milk sample",
            "positive_text": "Automated flush cycle: Run cleaning protocol C-3 with 0.1M alkaline detergent at 50 deg C for 3 minutes, followed by sterile water flush.",
            "negative_text": "Conveyor roller bearing lubrication schedule for cold storage facilities.",
        },
    ]

    out_dir = BASE_DIR / "ml_models" / "checkpoints" / "bge_small_synthetic_smoke"
    base_revision = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

    print(f"Output directory: {out_dir}")
    print(f"Base revision: {base_revision}")
    print(f"Train examples: {len(synthetic_train)}, Val examples: {len(synthetic_val)}")

    report = finetune(
        train_rows=synthetic_train,
        val_rows=synthetic_val,
        out_dir=out_dir,
        base_revision=base_revision,
        epochs=1,
        batch_size=4,
        lr=2e-5,
        max_seq_length=256,
        seed=42,
        synthetic_smoke_test=True,
    )

    print("\n--- TRAINING EXECUTION REPORT ---")
    print(json.dumps(report, indent=2))

    # Register checkpoint in registry
    registered_profile = register_finetuned(
        path=report["checkpoint"],
        key="bge-small-synthetic-smoke",
        promotion_eligible=False,
        training_status="SYNTHETIC_SMOKE_TEST_INELIGIBLE_FOR_PRODUCTION",
        dataset_manifest_hash="SYNTHETIC_NO_COMPANY_DATA",
        base_revision=base_revision,
        checkpoint_hash=report.get("checkpoint_sha256", ""),
    )

    print(f"\nProfile registered: key={registered_profile.key}")
    print(f"Collection name: {registered_profile.collection}")
    print(f"Promotion eligible: {registered_profile.promotion_eligible}")
    print(f"Training status: {registered_profile.training_status}")

    # Verify reloading through get_profile / load_model
    p = get_profile("bge-small-synthetic-smoke")
    loaded = load_model(p)
    emb = loaded.encode(["Diagnostic query smoke check"], normalize_embeddings=True)
    print(f"Reload verified successfully! Embedding shape: {emb.shape}")

if __name__ == "__main__":
    main()
