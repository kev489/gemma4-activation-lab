from __future__ import annotations

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from gemma4_activation_lab.activation_datasets import (
    load_activation_examples,
    load_impactbench_record_index,
    resolve_activation_example,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "impactbench_autonomy" / "activation_examples" / "v1"
TRANSCRIPT_DIR = ROOT / "impactBench" / "impactbench_transcripts" / "all_models"


class ActivationDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.examples_path = DATASET_DIR / "examples.jsonl"
        cls.examples = load_activation_examples(cls.examples_path)
        cls.record_index = load_impactbench_record_index(
            TRANSCRIPT_DIR,
            transcript_models={row["transcript_model"] for row in cls.examples},
        )

    def test_manifest_matches_examples(self) -> None:
        manifest = json.loads((DATASET_DIR / "manifest.json").read_text(encoding="utf-8"))
        payload = self.examples_path.read_bytes()
        self.assertEqual(manifest["examples"]["rows"], len(self.examples))
        self.assertEqual(manifest["files"]["examples"]["bytes"], len(payload))
        self.assertEqual(
            manifest["files"]["examples"]["sha256"],
            hashlib.sha256(payload).hexdigest(),
        )
        self.assertEqual(
            manifest["examples"]["harmful_counts"],
            dict(
                sorted(
                    Counter(str(row["harmful"]).lower() for row in self.examples).items()
                )
            ),
        )

        review_path = DATASET_DIR / "excluded_or_review.jsonl"
        review_payload = review_path.read_bytes()
        review_rows = [
            json.loads(line)
            for line in review_payload.decode("utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(manifest["excluded_or_review"]["rows"], len(review_rows))
        self.assertEqual(manifest["files"]["excluded_or_review"]["bytes"], len(review_payload))
        self.assertEqual(
            manifest["files"]["excluded_or_review"]["sha256"],
            hashlib.sha256(review_payload).hexdigest(),
        )
        self.assertFalse(
            {row["example_id"] for row in self.examples}
            & {row["review_id"] for row in review_rows}
        )

    def test_all_examples_resolve_to_source(self) -> None:
        for example in self.examples:
            resolved = resolve_activation_example(example, self.record_index)
            pooling = example["pooling"]
            self.assertEqual(example["harmful"], resolved.record["harmful"])
            self.assertEqual(
                resolved.assistant_turn.text[
                    pooling["pooling_span_char_start"] : pooling["pooling_span_char_end"]
                ],
                pooling["assistant_pooling_text"],
            )


if __name__ == "__main__":
    unittest.main()
