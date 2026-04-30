"""Literature source influence analysis for second-iteration LPBF results.

This script ranks original literature sources by leave-one-source-out influence
on the model prediction and constrained qNEHVI score of the final two rows in
``data_train_2.xlsx``.
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils.data_load import (
    DEFAULT_BASELINE,
    Get_stand,
    apply_fidelity_to_labels,
    compute_fidelity,
    compute_total_cost,
    extract_features,
    extract_labels,
    extract_process_info,
)


DEFAULT_LITERATURE_ROWS = 138
DEFAULT_TARGET_COUNT = 2
DEFAULT_TOP_K = 5
DEFAULT_OUTPUT_DIR = "analysis_outputs/source_influence"

KEYWORDS = [
    "melt pool",
    "porosity",
    "lack of fusion",
    "keyhole",
    "microstructure",
    "alpha",
    "beta",
    "elongation",
    "tensile strength",
]


class LiteratureMappingError(ValueError):
    """Raised when PDF filename ranges cannot map cleanly to data rows."""


def build_literature_mapping_diagnostics(
    literature_dir, literature_rows=DEFAULT_LITERATURE_ROWS
):
    """Return row-level diagnostics for PDF filename to data-row mapping."""
    literature_dir = Path(literature_dir)
    records = []
    row_hits = {}

    for pdf_path in sorted(literature_dir.glob("*.pdf")):
        try:
            rows = parse_literature_filename(pdf_path)
        except ValueError as exc:
            records.append(
                {
                    "data_row": "",
                    "source_id": pdf_path.stem,
                    "pdf_file": pdf_path.name,
                    "problem": "invalid_filename",
                    "detail": str(exc),
                }
            )
            continue
        for row in rows:
            row_hits.setdefault(row, []).append(pdf_path.name)

    for row in range(1, literature_rows + 1):
        hits = row_hits.get(row, [])
        if not hits:
            records.append(
                {
                    "data_row": row,
                    "source_id": "",
                    "pdf_file": "",
                    "problem": "missing",
                    "detail": "No PDF filename maps to this literature row.",
                }
            )
        elif len(hits) > 1:
            records.append(
                {
                    "data_row": row,
                    "source_id": "",
                    "pdf_file": ";".join(hits),
                    "problem": "overlap",
                    "detail": "Multiple PDF filenames map to this literature row.",
                }
            )

    for row, hits in sorted(row_hits.items()):
        if row < 1 or row > literature_rows:
            records.append(
                {
                    "data_row": row,
                    "source_id": "",
                    "pdf_file": ";".join(hits),
                    "problem": "out_of_range",
                    "detail": "PDF filename maps outside the configured literature rows.",
                }
            )

    return pd.DataFrame.from_records(
        records, columns=["data_row", "source_id", "pdf_file", "problem", "detail"]
    )


def parse_literature_filename(path):
    """Parse a PDF filename like ``1.pdf`` or ``2-8.pdf`` into data rows."""
    name = Path(path).name
    match = re.match(r"^(\d+)(?:-(\d+))?\.pdf$", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError("Unsupported literature PDF filename: {}".format(name))

    start = int(match.group(1))
    end = int(match.group(2) or start)
    if start <= 0 or end < start:
        raise ValueError("Invalid literature PDF range: {}".format(name))
    return list(range(start, end + 1))


def discover_literature_sources(literature_dir, literature_rows=DEFAULT_LITERATURE_ROWS):
    """Build source metadata from numbered PDF files and validate coverage."""
    literature_dir = Path(literature_dir)
    if not literature_dir.exists():
        raise FileNotFoundError("Literature directory not found: {}".format(literature_dir))

    sources = []
    row_to_source = {}
    overlaps = []
    out_of_range = []

    for pdf_path in sorted(literature_dir.glob("*.pdf")):
        rows = parse_literature_filename(pdf_path)
        source_id = pdf_path.stem
        for row in rows:
            if row < 1 or row > literature_rows:
                out_of_range.append((row, source_id))
                continue
            if row in row_to_source:
                overlaps.append((row, row_to_source[row]["source_id"], source_id))
            row_to_source[row] = {
                "source_id": source_id,
                "pdf_file": pdf_path.name,
                "source_start": min(rows),
                "source_end": max(rows),
            }
        sources.append(
            {
                "source_id": source_id,
                "pdf_file": pdf_path.name,
                "rows": [row for row in rows if 1 <= row <= literature_rows],
                "source_start": min(rows),
                "source_end": max(rows),
            }
        )

    missing = [row for row in range(1, literature_rows + 1) if row not in row_to_source]
    if missing or overlaps or out_of_range:
        message = ["Invalid literature mapping."]
        if missing:
            message.append("Missing rows: {}".format(_compact_ranges(missing)))
        if overlaps:
            message.append("Overlaps: {}".format(overlaps[:10]))
        if out_of_range:
            message.append("Out-of-range rows: {}".format(out_of_range[:10]))
        error = LiteratureMappingError(" ".join(message))
        error.diagnostics = build_literature_mapping_diagnostics(
            literature_dir, literature_rows=literature_rows
        )
        raise error

    return sources, row_to_source


def _compact_ranges(values):
    values = sorted(values)
    if not values:
        return ""
    ranges = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append("{}-{}".format(start, prev) if start != prev else str(start))
        start = prev = value
    ranges.append("{}-{}".format(start, prev) if start != prev else str(start))
    return ",".join(ranges)


def build_row_source_map(data, row_to_source, literature_rows=DEFAULT_LITERATURE_ROWS):
    """Return a data-row to source map for all rows in the Excel table."""
    records = []
    for data_row in range(1, len(data) + 1):
        if data_row <= literature_rows:
            meta = row_to_source[data_row]
            row_kind = "literature"
        else:
            meta = {
                "source_id": "experiment_iteration",
                "pdf_file": "",
                "source_start": "",
                "source_end": "",
            }
            row_kind = "experiment"
        records.append(
            {
                "data_row": data_row,
                "source_id": meta["source_id"],
                "pdf_file": meta["pdf_file"],
                "source_start": meta["source_start"],
                "source_end": meta["source_end"],
                "row_kind": row_kind,
            }
        )
    return pd.DataFrame.from_records(records)


def split_train_and_targets(data, target_count=DEFAULT_TARGET_COUNT):
    """Split final target rows away from attribution training rows."""
    if len(data) <= target_count:
        raise ValueError("Data must contain more rows than target_count.")
    return data.iloc[:-target_count].copy(), data.iloc[-target_count:].copy()


def _source_rows_for_training(source, train_df):
    row_set = set(source["rows"])
    return [idx for idx in train_df.index if (idx + 1) in row_set]


def _standardize_frames(train_features, target_features):
    scaler_fea = Get_stand()
    scaler_fea.get_scaler("MinMaxScaler")
    all_features = pd.concat([train_features, target_features], axis=0)
    stand_features = scaler_fea.star_stand(all_features)
    split_idx = len(train_features)
    return scaler_fea, stand_features.iloc[:split_idx], stand_features.iloc[split_idx:]


def evaluate_targets_with_bo(
    train_df,
    target_df,
    literature_rows=DEFAULT_LITERATURE_ROWS,
    baseline=None,
    seed=0,
):
    """Fit BO surrogate models and evaluate target predictions/acquisition."""
    import torch

    from utils.model import (
        build_qnehvi_policy,
        evaluate_acquisition,
        fit_MultiFidelity_gp_model,
        fit_stgp_model,
    )

    if baseline is None:
        baseline = DEFAULT_BASELINE

    torch.manual_seed(seed)
    torch.set_default_dtype(torch.double)

    train_features = extract_features(train_df)
    target_features = extract_features(target_df)
    train_labels_raw = extract_labels(train_df)
    process_info = extract_process_info(train_df)
    data_rows = np.asarray(train_df.index, dtype=int) + 1

    fidelity = compute_fidelity(process_info, baseline=baseline)
    experiment_mask = data_rows > literature_rows
    fidelity_values = fidelity.to_numpy(dtype=float)
    fidelity_values[experiment_mask] = 1.0

    train_labels = apply_fidelity_to_labels(
        train_labels_raw,
        fidelity_values,
        data_rows=data_rows,
        literature_rows=literature_rows,
    )
    total_cost = compute_total_cost(train_features)["Total_Cost"]

    scaler_fea, train_fea_scaled, target_fea_scaled = _standardize_frames(
        train_features, target_features
    )
    scaler_lab = Get_stand()
    scaler_lab.get_scaler("StandardScaler")
    train_y_scaled = scaler_lab.star_stand(train_labels)

    train_x = torch.tensor(train_fea_scaled.to_numpy(dtype=float), dtype=torch.double)
    target_x = torch.tensor(target_fea_scaled.to_numpy(dtype=float), dtype=torch.double)
    train_y = torch.tensor(train_y_scaled.to_numpy(dtype=float), dtype=torch.double)
    train_fidelity = torch.tensor(fidelity_values, dtype=torch.double).view(-1, 1)
    train_x_full = torch.cat([train_x, train_fidelity], dim=1)
    target_x_full = torch.cat(
        [target_x, torch.ones(target_x.shape[0], 1, dtype=torch.double)], dim=1
    )

    model, _ = fit_MultiFidelity_gp_model(
        train_x_full, train_y, data_fidelity=train_x_full.shape[-1] - 1
    )
    cost_y = torch.tensor(
        total_cost.to_numpy(dtype=float).reshape(-1, 1), dtype=torch.double
    )
    constraint_model, _ = fit_stgp_model(train_x, cost_y)

    ref_point = torch.tensor(
        [train_y[:, 0].min(), train_y[:, 1].min()], dtype=torch.double
    )
    policy = build_qnehvi_policy(model, ref_point, train_y, train_x_full)

    with torch.no_grad():
        posterior = model.posterior(target_x_full)
        pred_mean_scaled = posterior.mean.detach().cpu().numpy()
        pred_var = posterior.variance.detach().cpu().numpy()
        pred_mean = scaler_lab.end_stand(pred_mean_scaled)
        utility_mean = (
            constraint_model.posterior(target_x).mean.detach().cpu().numpy().reshape(-1)
        )

    qnehvi = evaluate_acquisition(policy, target_x_full)
    constrained_qnehvi = qnehvi * utility_mean

    label_columns = [str(col) for col in train_labels_raw.columns]
    return {
        "prediction_mean": pred_mean,
        "prediction_variance": pred_var,
        "qnehvi": qnehvi,
        "utility_mean": utility_mean,
        "constrained_qnehvi": constrained_qnehvi,
        "uts_mean": float(np.mean(pred_mean[:, 0])),
        "te_mean": float(np.mean(pred_mean[:, 1])),
        "qnehvi_mean": float(np.mean(constrained_qnehvi)),
        "label_columns": label_columns,
        "feature_columns": [str(col) for col in train_features.columns],
    }


def score_influence_dataframe(df):
    """Normalize signed components and compute the composite contribution score."""
    scored = df.copy()
    for column in ["uts_contribution", "te_contribution", "qnehvi_contribution"]:
        denom = float(np.nanmax(np.abs(scored[column].to_numpy(dtype=float))))
        if not math.isfinite(denom) or denom == 0.0:
            denom = 1.0
        scored[column + "_norm"] = scored[column] / denom

    scored["composite_score"] = (
        0.4 * scored["uts_contribution_norm"]
        + 0.4 * scored["te_contribution_norm"]
        + 0.2 * scored["qnehvi_contribution_norm"]
    )
    return scored.sort_values("composite_score", ascending=False).reset_index(drop=True)


def compute_feature_distance(source_df, target_df):
    """Compute scaled Euclidean distance between source and target feature means."""
    source_features = extract_features(source_df).astype(float)
    target_features = extract_features(target_df).astype(float)
    combined = pd.concat([source_features, target_features], axis=0)
    span = combined.max() - combined.min()
    span = span.replace(0, 1.0)
    delta = (source_features.mean() - target_features.mean()) / span
    return float(np.sqrt(np.sum(np.square(delta.to_numpy(dtype=float)))))


def summarize_source_features(source_df):
    """Create min/max summary values for source process features."""
    features = extract_features(source_df).astype(float)
    summary = {}
    for column in features.columns:
        name = str(column)
        summary[name + "_min"] = float(features[column].min())
        summary[name + "_max"] = float(features[column].max())
        summary[name + "_mean"] = float(features[column].mean())
    return summary


def run_source_influence(
    data_path,
    literature_dir,
    output_dir=DEFAULT_OUTPUT_DIR,
    literature_rows=DEFAULT_LITERATURE_ROWS,
    target_count=DEFAULT_TARGET_COUNT,
    top_k=DEFAULT_TOP_K,
    max_sources=None,
    seed=0,
    evaluator=None,
):
    data_path = Path(data_path)
    literature_dir = Path(literature_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_excel(str(data_path))
    train_df, target_df = split_train_and_targets(data, target_count=target_count)
    try:
        sources, row_to_source = discover_literature_sources(
            literature_dir, literature_rows=literature_rows
        )
    except LiteratureMappingError as exc:
        diagnostics = getattr(exc, "diagnostics", None)
        if diagnostics is not None:
            diagnostics.to_csv(
                output_dir / "row_source_mapping_diagnostics.csv", index=False
            )
        raise
    row_source_map = build_row_source_map(
        data, row_to_source, literature_rows=literature_rows
    )
    row_source_map.to_csv(output_dir / "row_source_map.csv", index=False)

    target_points = target_df.copy()
    target_points.insert(0, "data_row", [idx + 1 for idx in target_points.index])
    target_points.to_csv(output_dir / "target_points.csv", index=False)

    candidate_sources = sources[:max_sources] if max_sources else sources
    influence = compute_source_influence(
        data,
        train_df,
        target_df,
        candidate_sources,
        literature_rows=literature_rows,
        seed=seed,
        evaluator=evaluator,
    )
    influence.to_csv(output_dir / "literature_influence.csv", index=False)
    write_mechanism_report(
        output_dir / "top_literature_mechanism.md",
        influence,
        data,
        target_df,
        literature_dir,
        top_k=top_k,
    )
    return influence


def compute_source_influence(
    data,
    train_df,
    target_df,
    sources,
    literature_rows=DEFAULT_LITERATURE_ROWS,
    seed=0,
    evaluator=None,
):
    """Run the leave-one-source-out attribution loop."""
    if evaluator is None:
        evaluator = evaluate_targets_with_bo

    baseline_metrics = evaluator(
        train_df, target_df, literature_rows=literature_rows, seed=seed
    )

    influence_records = []
    for source in sources:
        remove_indices = _source_rows_for_training(source, train_df)
        if not remove_indices:
            continue
        leaveout_train = train_df.drop(index=remove_indices)
        leaveout_metrics = evaluator(
            leaveout_train, target_df, literature_rows=literature_rows, seed=seed
        )
        source_df = data.iloc[[row - 1 for row in source["rows"]]]
        record = {
            "source_id": source["source_id"],
            "pdf_file": source["pdf_file"],
            "source_start": source["source_start"],
            "source_end": source["source_end"],
            "n_rows": len(source["rows"]),
            "removed_data_rows": _compact_ranges(source["rows"]),
            "baseline_uts_mean": baseline_metrics["uts_mean"],
            "leaveout_uts_mean": leaveout_metrics["uts_mean"],
            "uts_contribution": baseline_metrics["uts_mean"]
            - leaveout_metrics["uts_mean"],
            "baseline_te_mean": baseline_metrics["te_mean"],
            "leaveout_te_mean": leaveout_metrics["te_mean"],
            "te_contribution": baseline_metrics["te_mean"]
            - leaveout_metrics["te_mean"],
            "baseline_qnehvi_mean": baseline_metrics["qnehvi_mean"],
            "leaveout_qnehvi_mean": leaveout_metrics["qnehvi_mean"],
            "qnehvi_contribution": baseline_metrics["qnehvi_mean"]
            - leaveout_metrics["qnehvi_mean"],
            "feature_distance_to_targets": compute_feature_distance(
                source_df, target_df
            ),
        }
        record.update(summarize_source_features(source_df))
        influence_records.append(record)

    if not influence_records:
        raise ValueError("No literature sources overlapped the training data.")
    return score_influence_dataframe(pd.DataFrame.from_records(influence_records))


def _format_float(value, digits=4):
    try:
        if value is None or not math.isfinite(float(value)):
            return ""
        return ("{:.%df}" % digits).format(float(value))
    except Exception:
        return str(value)


def _markdown_table(rows, headers):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def _keyword_hits(pdf_path):
    try:
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader
    except ImportError:
        return None, "No PDF text extractor installed."

    try:
        reader = PdfReader(str(pdf_path))
        text_parts = []
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                continue
        text = "\n".join(text_parts).lower()
    except Exception as exc:
        return None, "PDF text extraction failed: {}".format(exc)

    hits = {}
    for keyword in KEYWORDS:
        hits[keyword] = text.count(keyword.lower())
    return hits, None


def _mechanism_note(source_df, target_df):
    source_features = extract_features(source_df).astype(float)
    target_features = extract_features(target_df).astype(float)
    feature_names = [str(col) for col in source_features.columns]
    notes = []

    for column in source_features.columns:
        source_min = float(source_features[column].min())
        source_max = float(source_features[column].max())
        target_min = float(target_features[column].min())
        target_max = float(target_features[column].max())
        if source_min <= target_max and target_min <= source_max:
            notes.append(
                "{} overlaps the target window, so this source likely anchors the "
                "local process-property response.".format(str(column))
            )

    if not notes:
        notes.append(
            "The source is not closest by every raw feature, so its contribution is "
            "likely coming through learned multi-objective trends rather than simple "
            "nearest-neighbor similarity."
        )

    if any("energy" in name.lower() or "ved" in name.lower() for name in feature_names):
        notes.append(
            "VED alignment should be checked first for melt-pool stability, lack-of-"
            "fusion suppression, keyhole avoidance, and strength-ductility balance."
        )
    else:
        notes.append(
            "Compare power-speed balance against the target point to assess melt-pool "
            "stability and defect control mechanisms."
        )

    return " ".join(notes)


def write_mechanism_report(
    report_path, influence, data, target_df, literature_dir, top_k=DEFAULT_TOP_K
):
    top = influence.head(top_k)
    target_display = target_df.copy()
    target_display.insert(0, "data_row", [idx + 1 for idx in target_df.index])

    lines = []
    lines.append("# Top Literature Mechanism Summary")
    lines.append("")
    lines.append("## Target Points")
    target_rows = []
    for _, row in target_display.iterrows():
        target_rows.append({str(col): _format_float(row[col]) for col in target_display.columns})
    lines.append(_markdown_table(target_rows, [str(col) for col in target_display.columns]))
    lines.append("")
    lines.append("## Top Literature Influence")
    summary_rows = []
    for _, row in top.iterrows():
        summary_rows.append(
            {
                "rank": len(summary_rows) + 1,
                "source_id": row["source_id"],
                "pdf_file": row["pdf_file"],
                "rows": row["removed_data_rows"],
                "score": _format_float(row["composite_score"]),
                "uts_delta": _format_float(row["uts_contribution"]),
                "te_delta": _format_float(row["te_contribution"]),
                "qnehvi_delta": _format_float(row["qnehvi_contribution"]),
                "distance": _format_float(row["feature_distance_to_targets"]),
            }
        )
    lines.append(
        _markdown_table(
            summary_rows,
            [
                "rank",
                "source_id",
                "pdf_file",
                "rows",
                "score",
                "uts_delta",
                "te_delta",
                "qnehvi_delta",
                "distance",
            ],
        )
    )
    lines.append("")
    lines.append("## Mechanism Notes")

    literature_dir = Path(literature_dir)
    for _, row in top.iterrows():
        source_rows = [int(x) for x in str(row["removed_data_rows"]).replace(",", "-").split("-") if x]
        if len(source_rows) >= 2:
            row_range = range(min(source_rows), max(source_rows) + 1)
        else:
            row_range = source_rows
        source_df = data.iloc[[data_row - 1 for data_row in row_range]]
        pdf_path = literature_dir / row["pdf_file"]
        hits, error = _keyword_hits(pdf_path)

        lines.append("")
        lines.append("### {} ({})".format(row["source_id"], row["pdf_file"]))
        lines.append("- Rows: {}".format(row["removed_data_rows"]))
        lines.append("- Composite score: {}".format(_format_float(row["composite_score"])))
        lines.append("- Parameter distance to targets: {}".format(_format_float(row["feature_distance_to_targets"])))
        lines.append("- Interpretation: {}".format(_mechanism_note(source_df, target_df)))
        if hits is None:
            lines.append("- PDF keyword scan: {}".format(error))
        else:
            nonzero = ["{}={}".format(k, v) for k, v in hits.items() if v > 0]
            lines.append("- PDF keyword scan: {}".format(", ".join(nonzero) if nonzero else "no configured keyword hits"))

    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/data_train_2.xlsx")
    parser.add_argument("--literature-dir", default="original_literature")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--literature-rows", type=int, default=DEFAULT_LITERATURE_ROWS)
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    influence = run_source_influence(
        data_path=args.data,
        literature_dir=args.literature_dir,
        output_dir=args.output_dir,
        literature_rows=args.literature_rows,
        target_count=args.target_count,
        top_k=args.top_k,
        max_sources=args.max_sources,
        seed=args.seed,
    )
    print("Wrote {} literature influence rows.".format(len(influence)))
    print("Outputs are in {}".format(args.output_dir))


if __name__ == "__main__":
    main()
