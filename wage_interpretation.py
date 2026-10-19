from __future__ import annotations

import numpy as np
import pandas as pd


def _require_frame(frame, name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")


def _require_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _normalize_dates(frame: pd.DataFrame, date_columns: list[str], name: str) -> pd.DataFrame:
    normalized = frame.copy()
    for column in date_columns:
        values = pd.to_datetime(normalized[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{name}.{column} contains invalid dates")
        normalized[column] = values.dt.normalize()
    return normalized


def _require_finite_numeric(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"{name}.{column} contains non-finite numeric values")


def build_interpretation_summary(
    decomposition_summary,
    fixed_composition,
    end_date,
) -> pd.DataFrame:
    _require_frame(decomposition_summary, "decomposition_summary")
    _require_frame(fixed_composition, "fixed_composition")

    decomp_required = [
        "base_date",
        "date",
        "published_change",
        "reconstructed_change",
        "within_total",
        "composition_total",
        "decomposition_residual",
        "benchmark_gap",
    ]
    fixed_required = [
        "date",
        "reference_date",
        "fixed_composition_ahe",
        "reconstructed_ahe",
        "published_total_private_ahe",
    ]
    _require_columns(decomposition_summary, decomp_required, "decomposition_summary")
    _require_columns(fixed_composition, fixed_required, "fixed_composition")

    decomp = _normalize_dates(
        decomposition_summary, ["base_date", "date"], "decomposition_summary"
    )
    fixed = _normalize_dates(fixed_composition, ["date", "reference_date"], "fixed_composition")

    end_ts = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(end_ts):
        raise ValueError("end_date is invalid")
    end_ts = end_ts.normalize()

    decomp_match = decomp[decomp["date"] == end_ts]
    fixed_match = fixed[fixed["date"] == end_ts]

    if decomp_match.empty:
        raise ValueError(f"decomposition_summary is missing end_date {end_ts}")
    if len(decomp_match) > 1:
        raise ValueError(f"decomposition_summary has duplicate rows for end_date {end_ts}")
    if fixed_match.empty:
        raise ValueError(
            f"fixed_composition endpoint date mismatch for end_date {end_ts}: missing row"
        )
    if len(fixed_match) > 1:
        raise ValueError(f"fixed_composition has duplicate rows for end_date {end_ts}")

    decomp_row = decomp_match.iloc[0]
    fixed_row = fixed_match.iloc[0]

    if fixed_row["reference_date"] != decomp_row["base_date"]:
        raise ValueError(
            "reference_date/base_date mismatch for selected end_date interval"
        )

    decomp_numeric = [
        "published_change",
        "reconstructed_change",
        "within_total",
        "composition_total",
        "decomposition_residual",
        "benchmark_gap",
    ]
    fixed_numeric = [
        "fixed_composition_ahe",
        "reconstructed_ahe",
        "published_total_private_ahe",
    ]
    _require_finite_numeric(decomp_match, decomp_numeric, "decomposition_summary")
    _require_finite_numeric(fixed_match, fixed_numeric, "fixed_composition")

    return pd.DataFrame(
        [
            {
                "base_date": decomp_row["base_date"],
                "end_date": decomp_row["date"],
                "reference_date": fixed_row["reference_date"],
                "published_change": decomp_row["published_change"],
                "reconstructed_change": decomp_row["reconstructed_change"],
                "within_total": decomp_row["within_total"],
                "composition_total": decomp_row["composition_total"],
                "decomposition_residual": decomp_row["decomposition_residual"],
                "benchmark_gap": decomp_row["benchmark_gap"],
                "fixed_composition_ahe": fixed_row["fixed_composition_ahe"],
                "reconstructed_ahe": fixed_row["reconstructed_ahe"],
                "published_ahe": fixed_row["published_total_private_ahe"],
            }
        ],
        columns=[
            "base_date",
            "end_date",
            "reference_date",
            "published_change",
            "reconstructed_change",
            "within_total",
            "composition_total",
            "decomposition_residual",
            "benchmark_gap",
            "fixed_composition_ahe",
            "reconstructed_ahe",
            "published_ahe",
        ],
    )


def summarize_composition_materiality(
    interpretation_summary,
    contributions,
    end_date,
) -> pd.DataFrame:
    _require_frame(interpretation_summary, "interpretation_summary")
    _require_frame(contributions, "contributions")

    _require_columns(
        interpretation_summary,
        ["base_date", "end_date", "reconstructed_change", "composition_total"],
        "interpretation_summary",
    )
    _require_columns(
        contributions,
        ["base_date", "date", "industry_code", "composition_contribution"],
        "contributions",
    )

    summary = _normalize_dates(
        interpretation_summary, ["base_date", "end_date"], "interpretation_summary"
    )
    contrib = _normalize_dates(contributions, ["base_date", "date"], "contributions")

    end_ts = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(end_ts):
        raise ValueError("end_date is invalid")
    end_ts = end_ts.normalize()

    if contrib.duplicated(subset=["base_date", "date", "industry_code"]).any():
        raise ValueError(
            "contributions has duplicate (base_date, date, industry_code) rows"
        )

    summary_match = summary[summary["end_date"] == end_ts]
    if summary_match.empty:
        raise ValueError(f"interpretation_summary is missing end_date {end_ts}")
    if len(summary_match) > 1:
        raise ValueError(f"interpretation_summary has duplicate rows for end_date {end_ts}")

    summary_row = summary_match.iloc[0]
    base_ts = summary_row["base_date"]

    contrib_end = contrib[contrib["date"] == end_ts]
    if contrib_end.empty:
        raise ValueError(f"contributions is missing end_date {end_ts}")
    if (contrib_end["base_date"] != base_ts).any():
        raise ValueError(
            "contributions endpoint/base_date mismatch for selected end_date"
        )

    interval = contrib_end[contrib_end["base_date"] == base_ts].copy()
    if interval.empty:
        raise ValueError(
            "contributions endpoint/base_date mismatch for selected end_date"
        )

    _require_finite_numeric(
        summary_match,
        ["reconstructed_change", "composition_total"],
        "interpretation_summary",
    )
    _require_finite_numeric(
        interval,
        ["composition_contribution"],
        "contributions",
    )

    composition_total = float(summary_row["composition_total"])
    reconstructed_change = float(summary_row["reconstructed_change"])

    positive_total = float(
        interval.loc[interval["composition_contribution"] > 0, "composition_contribution"].sum()
    )
    negative_total = float(
        interval.loc[interval["composition_contribution"] < 0, "composition_contribution"].sum()
    )
    gross_abs = float(interval["composition_contribution"].abs().sum())
    computed_net = float(interval["composition_contribution"].sum())

    if not np.isclose(computed_net, composition_total, rtol=0.0, atol=1e-9):
        raise ValueError(
            "composition sum mismatch with interpretation_summary composition_total beyond tolerance"
        )

    net_to_gross_ratio = np.nan if np.isclose(gross_abs, 0.0) else composition_total / gross_abs
    share_of_reconstructed = (
        np.nan
        if np.isclose(reconstructed_change, 0.0)
        else composition_total / reconstructed_change
    )

    return pd.DataFrame(
        [
            {
                "base_date": base_ts,
                "end_date": end_ts,
                "composition_total": composition_total,
                "gross_abs_composition": gross_abs,
                "positive_composition_total": positive_total,
                "negative_composition_total": negative_total,
                "net_to_gross_ratio": net_to_gross_ratio,
                "composition_share_of_reconstructed_change": share_of_reconstructed,
            }
        ],
        columns=[
            "base_date",
            "end_date",
            "composition_total",
            "gross_abs_composition",
            "positive_composition_total",
            "negative_composition_total",
            "net_to_gross_ratio",
            "composition_share_of_reconstructed_change",
        ],
    )


def build_composition_contributor_tables(
    contributions,
    *,
    base_date,
    end_date,
    top_n=3,
) -> dict[str, pd.DataFrame]:
    _require_frame(contributions, "contributions")
    _require_columns(
        contributions,
        ["base_date", "date", "industry_code", "composition_contribution"],
        "contributions",
    )

    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")

    base_ts = pd.to_datetime(base_date, errors="coerce")
    end_ts = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(base_ts):
        raise ValueError("base_date is invalid")
    if pd.isna(end_ts):
        raise ValueError("end_date is invalid")
    base_ts = base_ts.normalize()
    end_ts = end_ts.normalize()

    contrib = _normalize_dates(contributions, ["base_date", "date"], "contributions")
    if contrib.duplicated(subset=["base_date", "date", "industry_code"]).any():
        raise ValueError(
            "contributions has duplicate (base_date, date, industry_code) rows"
        )
    _require_finite_numeric(contrib, ["composition_contribution"], "contributions")

    interval = contrib[
        (contrib["base_date"] == base_ts) & (contrib["date"] == end_ts)
    ].copy()
    if interval.empty:
        raise ValueError(
            f"missing interval for base_date={base_ts} and end_date={end_ts}"
        )

    interval["absolute_contribution"] = interval["composition_contribution"].abs()

    all_table = interval.sort_values(
        ["absolute_contribution", "industry_code"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    positive_table = all_table[all_table["composition_contribution"] > 0].copy()
    positive_table = positive_table.sort_values(
        ["composition_contribution", "industry_code"],
        ascending=[False, True],
        kind="mergesort",
    ).head(top_n).reset_index(drop=True)

    negative_table = all_table[all_table["composition_contribution"] < 0].copy()
    negative_table = negative_table.sort_values(
        ["composition_contribution", "industry_code"],
        ascending=[True, True],
        kind="mergesort",
    ).head(top_n).reset_index(drop=True)

    return {
        "all": all_table,
        "positive": positive_table,
        "negative": negative_table,
    }


__all__ = [
    "build_interpretation_summary",
    "summarize_composition_materiality",
    "build_composition_contributor_tables",
]
