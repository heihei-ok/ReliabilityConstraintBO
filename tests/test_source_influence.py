import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.source_influence import (
    LiteratureMappingError,
    build_literature_mapping_diagnostics,
    build_row_source_map,
    compute_source_influence,
    discover_literature_sources,
    parse_literature_filename,
    score_influence_dataframe,
    split_train_and_targets,
)


class SourceInfluenceTests(unittest.TestCase):
    def test_parse_single_and_range_pdf_names(self):
        self.assertEqual(parse_literature_filename("1.pdf"), [1])
        self.assertEqual(parse_literature_filename("2-4.pdf"), [2, 3, 4])

    def test_parse_rejects_invalid_names(self):
        with self.assertRaises(ValueError):
            parse_literature_filename("paper-a.pdf")
        with self.assertRaises(ValueError):
            parse_literature_filename("8-2.pdf")

    def test_literature_mapping_full_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "1.pdf").write_bytes(b"")
            (root / "2-3.pdf").write_bytes(b"")
            sources, row_to_source = discover_literature_sources(root, literature_rows=3)
            self.assertEqual(len(sources), 2)
            self.assertEqual(row_to_source[1]["source_id"], "1")
            self.assertEqual(row_to_source[3]["source_id"], "2-3")

    def test_literature_mapping_reports_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "1.pdf").write_bytes(b"")
            (root / "3.pdf").write_bytes(b"")
            with self.assertRaises(LiteratureMappingError):
                discover_literature_sources(root, literature_rows=3)

    def test_literature_mapping_reports_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "1-2.pdf").write_bytes(b"")
            (root / "2-3.pdf").write_bytes(b"")
            with self.assertRaises(LiteratureMappingError):
                discover_literature_sources(root, literature_rows=3)
            diagnostics = build_literature_mapping_diagnostics(root, literature_rows=3)
            self.assertIn("overlap", diagnostics["problem"].tolist())

    def test_row_source_map_marks_experiment_rows(self):
        data = pd.DataFrame({"x": range(5)})
        row_to_source = {
            1: {"source_id": "1", "pdf_file": "1.pdf", "source_start": 1, "source_end": 1},
            2: {"source_id": "2", "pdf_file": "2.pdf", "source_start": 2, "source_end": 2},
            3: {"source_id": "3", "pdf_file": "3.pdf", "source_start": 3, "source_end": 3},
        }
        mapped = build_row_source_map(data, row_to_source, literature_rows=3)
        self.assertEqual(mapped.loc[0, "row_kind"], "literature")
        self.assertEqual(mapped.loc[3, "row_kind"], "experiment")
        self.assertEqual(mapped.loc[4, "source_id"], "experiment_iteration")

    def test_split_train_and_targets_uses_final_rows(self):
        data = pd.DataFrame({"x": range(5)})
        train, target = split_train_and_targets(data, target_count=2)
        self.assertEqual(train["x"].tolist(), [0, 1, 2])
        self.assertEqual(target["x"].tolist(), [3, 4])

    def test_score_influence_dataframe_preserves_signed_deltas(self):
        df = pd.DataFrame(
            {
                "source_id": ["a", "b"],
                "uts_contribution": [2.0, -1.0],
                "te_contribution": [1.0, 2.0],
                "qnehvi_contribution": [0.5, -0.5],
            }
        )
        scored = score_influence_dataframe(df)
        self.assertIn("composite_score", scored.columns)
        self.assertEqual(scored.iloc[0]["source_id"], "a")
        self.assertLess(scored.loc[scored["source_id"] == "b", "uts_contribution"].iloc[0], 0)

    def test_compute_source_influence_with_mock_evaluator(self):
        data = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "power": [100, 120, 140, 160, 180],
                "speed": [800, 900, 1000, 1100, 1200],
                "ved": [60, 63, 66, 69, 72],
                "uts": [900, 920, 940, 960, 980],
                "te": [10, 11, 12, 13, 14],
                "powder": ["a", "a", "a", "a", "a"],
                "size": [30, 30, 30, 30, 30],
            }
        )
        train, target = split_train_and_targets(data, target_count=1)
        sources = [
            {
                "source_id": "1",
                "pdf_file": "1.pdf",
                "source_start": 1,
                "source_end": 1,
                "rows": [1],
            },
            {
                "source_id": "2",
                "pdf_file": "2.pdf",
                "source_start": 2,
                "source_end": 2,
                "rows": [2],
            },
        ]

        def fake_evaluator(train_df, target_df, literature_rows=3, seed=0):
            return {
                "uts_mean": float(train_df["uts"].sum()),
                "te_mean": float(train_df["te"].sum()),
                "qnehvi_mean": float(len(train_df)),
            }

        influence = compute_source_influence(
            data,
            train,
            target,
            sources,
            literature_rows=3,
            evaluator=fake_evaluator,
        )
        self.assertEqual(set(influence["source_id"]), {"1", "2"})
        self.assertIn("composite_score", influence.columns)
        self.assertTrue((influence["uts_contribution"] > 0).all())


if __name__ == "__main__":
    unittest.main()
