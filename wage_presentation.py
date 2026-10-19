from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def _coerce_target_date(value, name: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"{name} is an invalid date")
    return ts.normalize()


def build_headline_comparison_chart_data(
    annual_summary,
    fixed_composition,
    *,
    base_date,
    end_date,
) -> pd.DataFrame:
    _require_frame(annual_summary, "annual_summary")
    _require_frame(fixed_composition, "fixed_composition")
    _require_columns(
        annual_summary,
        ["base_date", "date", "published_change", "reconstructed_change"],
        "annual_summary",
    )
    _require_columns(
        fixed_composition,
        ["date", "reference_date", "fixed_composition_ahe"],
        "fixed_composition",
    )

    summary = _normalize_dates(annual_summary, ["base_date", "date"], "annual_summary")
    fixed = _normalize_dates(
        fixed_composition, ["date", "reference_date"], "fixed_composition"
    )

    base_ts = _coerce_target_date(base_date, "base_date")
    end_ts = _coerce_target_date(end_date, "end_date")

    summary_match = summary[(summary["base_date"] == base_ts) & (summary["date"] == end_ts)]
    if summary_match.empty:
        raise ValueError(
            f"annual_summary missing row for base_date={base_ts} and end_date={end_ts}"
        )
    if len(summary_match) > 1:
        raise ValueError(
            f"annual_summary has duplicate rows for base_date={base_ts} and end_date={end_ts}"
        )
    _require_finite_numeric(
        summary_match, ["published_change", "reconstructed_change"], "annual_summary"
    )

    fixed_base = fixed[fixed["date"] == base_ts]
    fixed_end = fixed[fixed["date"] == end_ts]
    if fixed_base.empty:
        raise ValueError(f"fixed_composition missing base_date row for {base_ts}")
    if fixed_end.empty:
        raise ValueError(f"fixed_composition missing end_date row for {end_ts}")
    if len(fixed_base) > 1:
        raise ValueError(f"fixed_composition has duplicate base_date rows for {base_ts}")
    if len(fixed_end) > 1:
        raise ValueError(f"fixed_composition has duplicate end_date rows for {end_ts}")
    _require_finite_numeric(fixed_base, ["fixed_composition_ahe"], "fixed_composition")
    _require_finite_numeric(fixed_end, ["fixed_composition_ahe"], "fixed_composition")

    fixed_base_row = fixed_base.iloc[0]
    fixed_end_row = fixed_end.iloc[0]
    reference_date = fixed_end_row["reference_date"]
    if fixed_base_row["reference_date"] != reference_date:
        raise ValueError("fixed_composition reference_date mismatch across endpoints")
    if reference_date != base_ts:
        raise ValueError("fixed_composition reference_date/base_date mismatch")

    summary_row = summary_match.iloc[0]
    published = float(summary_row["published_change"])
    reconstructed = float(summary_row["reconstructed_change"])
    fixed_change = float(fixed_end_row["fixed_composition_ahe"]) - float(
        fixed_base_row["fixed_composition_ahe"]
    )

    rows = [
        {
            "base_date": base_ts,
            "end_date": end_ts,
            "reference_date": reference_date,
            "series": "published",
            "value": published,
            "unit": "dollars_per_hour_change",
        },
        {
            "base_date": base_ts,
            "end_date": end_ts,
            "reference_date": reference_date,
            "series": "reconstructed",
            "value": reconstructed,
            "unit": "dollars_per_hour_change",
        },
        {
            "base_date": base_ts,
            "end_date": end_ts,
            "reference_date": reference_date,
            "series": "fixed_composition",
            "value": fixed_change,
            "unit": "dollars_per_hour_change",
        },
    ]
    return pd.DataFrame(
        rows,
        columns=["base_date", "end_date", "reference_date", "series", "value", "unit"],
    )


def build_composition_contribution_chart_data(
    contributions,
    *,
    base_date,
    end_date,
) -> pd.DataFrame:
    _require_frame(contributions, "contributions")
    _require_columns(
        contributions,
        ["base_date", "date", "industry_code", "composition_contribution"],
        "contributions",
    )

    contrib = _normalize_dates(contributions, ["base_date", "date"], "contributions")
    base_ts = _coerce_target_date(base_date, "base_date")
    end_ts = _coerce_target_date(end_date, "end_date")

    if contrib.duplicated(subset=["base_date", "date", "industry_code"]).any():
        raise ValueError("contributions has duplicate (base_date, date, industry_code) rows")
    _require_finite_numeric(contrib, ["composition_contribution"], "contributions")

    interval = contrib[(contrib["base_date"] == base_ts) & (contrib["date"] == end_ts)].copy()
    if interval.empty:
        raise ValueError(
            f"contributions missing interval for base_date={base_ts} and end_date={end_ts}"
        )

    if "industry_name" not in interval.columns:
        interval["industry_name"] = pd.NA

    interval = interval.rename(columns={"date": "end_date"})
    interval["absolute_contribution"] = interval["composition_contribution"].abs()
    interval["direction"] = np.where(
        interval["composition_contribution"] > 0,
        "positive",
        np.where(interval["composition_contribution"] < 0, "negative", "zero"),
    )
    interval["label"] = interval["industry_name"]
    missing_label = interval["label"].isna() | (interval["label"].astype(str).str.len() == 0)
    interval.loc[missing_label, "label"] = interval.loc[missing_label, "industry_code"]

    interval = interval.sort_values(
        ["composition_contribution", "industry_code"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return interval[
        [
            "base_date",
            "end_date",
            "industry_code",
            "industry_name",
            "composition_contribution",
            "absolute_contribution",
            "direction",
            "label",
        ]
    ].copy()


def plot_headline_comparison_chart(chart_data, *, ax=None):
    _require_frame(chart_data, "chart_data")
    _require_columns(chart_data, ["series", "value"], "chart_data")
    if chart_data.empty:
        raise ValueError("chart_data has no rows")

    data = chart_data.copy()
    _require_finite_numeric(data, ["value"], "chart_data")

    if ax is None:
        _, ax = plt.subplots()

    ax.bar(data["series"], data["value"])
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_title("Recent Wage Growth: Headline vs Fixed Composition")
    ax.set_ylabel("Average hourly earnings change (dollars per hour)")
    ax.tick_params(axis="x", rotation=20)
    return ax


def plot_composition_contribution_chart(chart_data, *, ax=None):
    _require_frame(chart_data, "chart_data")
    _require_columns(chart_data, ["label", "composition_contribution"], "chart_data")
    if chart_data.empty:
        raise ValueError("chart_data has no rows")

    data = chart_data.copy()
    _require_finite_numeric(data, ["composition_contribution"], "chart_data")

    if ax is None:
        _, ax = plt.subplots()

    colors = np.where(
        data["composition_contribution"] > 0,
        "#2ca02c",
        np.where(data["composition_contribution"] < 0, "#d62728", "#7f7f7f"),
    )
    ax.barh(data["label"], data["composition_contribution"], color=colors)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_title("Industry Composition Contributions to Wage Growth")
    ax.set_xlabel("Composition contribution (dollars per hour)")
    return ax


__all__ = [
    "build_headline_comparison_chart_data",
    "build_composition_contribution_chart_data",
    "plot_headline_comparison_chart",
    "plot_composition_contribution_chart",
]
