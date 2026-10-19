"""Inspect a separate Qwen3 Chroma index before enabling experimental serving.

No model downloads, mutations, or company record deletions. Set
SERVY_QWEN3_INDEX_VERIFIED=1 *only* after complete=True is observed.
"""
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Verify the Qwen3 1024-D index is complete and stamped for its local model"

    def add_arguments(self, parser):
        parser.add_argument(
            "--profile", choices=("qwen3", "company-qwen3"),
            required=True,
            help="Use company-qwen3 only for authorized real-company data",
        )

    def handle(self, *args, **options):
        from core.ml.embeddings import get_profile
        from core.ml.indexer import verify_index

        profile = get_profile(options["profile"])
        if not profile.local_path:
            raise CommandError("SERVY_QWEN3_MODEL_PATH is required (offline model folder)")
        try:
            report = verify_index(profile.key)
        except Exception as exc:
            raise CommandError(f"Qwen3 index verification failed: {exc}") from exc
        self.stdout.write(
            "QWEN3 INDEX: " + str({
                "profile": profile.key, "collection": profile.collection,
                "eligible": report["expected"], "present": report["present"],
                "missing": report["missing"], "orphans": report["orphans"],
                "metadata_mismatch": report["metadata_mismatch"],
                "wrong_model_rows": report["wrong_model_rows"],
                "unresolvable_chunk_ids": report["unresolvable_chunk_ids"],
            })
        )
        if not report["complete"] or report["expected"] < 1:
            raise CommandError(
                "NOT VERIFIED: keep Qwen3 disabled. Repair index without "
                "deleting company data, then rerun."
            )
        self.stdout.write(self.style.SUCCESS(
            "QWEN3 INDEX VERIFIED: complete, identity-stamped and nonempty. "
            "Experimental flags are still off unless explicitly enabled."
        ))
