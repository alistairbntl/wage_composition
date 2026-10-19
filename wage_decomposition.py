from __future__ import annotations

import numpy as np
import pandas as pd


def _require_frame(frame, name):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")


def _require_columns(frame, required, name):
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _coerce_date_series(frame, name):
    df = frame.copy()
    _require_columns(df, ["date"], name)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _validate_component_frame(component_panel, base_date, end_date):
    _require_frame(component_panel, "component_panel")
    _require_columns(
        component_panel, ["date", "industry_code", "ahe"], "component_panel"
    )

    panel = _coerce_date_series(component_panel, "component_panel")

    if not (base_date < end_date):
        raise ValueError("base_date must be earlier than end_date")

    duplicates = panel.duplicated(subset=["date", "industry_code"], keep=False)
    if duplicates.any():
        bad = panel.loc[duplicates, ["date", "industry_code"]].drop_duplicates()
        raise ValueError(
            "component_panel contains duplicate (date, industry_code) rows: "
            f"{bad.to_dict(orient='records')}"
        )

    required_dates = {base_date, end_date}
    if not required_dates.issubset(set(pd.to_datetime(panel["date"]).unique())):
        raise ValueError(
            "component_panel missing required endpoint dates: "
            f"expected {sorted(pd.to_datetime(list(required_dates)))}"
        )

    ahe_numeric = pd.to_numeric(panel["ahe"], errors="coerce")
    if ahe_numeric.isna().any() or not np.isfinite(ahe_numeric.to_numpy()).all():
        raise ValueError(
            "component_panel contains missing, nonnumeric, or nonfinite ahe values"
        )

    return panel


def _validate_weights_frame(weights, base_date, end_date):
    _require_frame(weights, "weights")
    _require_columns(weights, ["date", "industry_code", "weight"], "weights")

    weight_df = _coerce_date_series(weights, "weights")

    duplicates = weight_df.duplicated(subset=["date", "industry_code"], keep=False)
    if duplicates.any():
        bad = weight_df.loc[duplicates, ["date", "industry_code"]].drop_duplicates()
        raise ValueError(
            "weights contains duplicate (date, industry_code) rows: "
            f"{bad.to_dict(orient='records')}"
        )

    required_dates = {base_date, end_date}
    if not required_dates.issubset(set(pd.to_datetime(weight_df["date"]).unique())):
        raise ValueError(
            "weights missing required endpoint dates: "
            f"expected {sorted(pd.to_datetime(list(required_dates)))}"
        )

    weight_numeric = pd.to_numeric(weight_df["weight"], errors="coerce")
    if weight_numeric.isna().any() or not np.isfinite(weight_numeric.to_numpy()).all():
        raise ValueError("weights contains missing, nonnumeric, or nonfinite values")
    if (weight_numeric < 0).any():
        raise ValueError("weights must be nonnegative")

    date_sums = weight_df.groupby("date")["weight"].sum()
    bad_dates = date_sums[~np.isclose(date_sums, 1.0)].index.tolist()
    if bad_dates:
        raise ValueError(
            "weights must sum to one for each date; invalid dates: "
            f"{sorted(pd.to_datetime(bad_dates))}"
        )

    return weight_df


def _normalize_benchmark_columns(frame):
    if "published_total_private_ahe" in frame.columns:
        return "published_total_private_ahe", None
    if "ahe" in frame.columns:
        return "ahe", None
    raise ValueError(
        "benchmark must contain 'published_total_private_ahe' or 'ahe' for the published total-private AHE"
    )


def _validate_benchmark_frame(benchmark, base_date, end_date):
    _require_frame(benchmark, "benchmark")
    _require_columns(benchmark, ["date"], "benchmark")

    bench = _coerce_date_series(benchmark, "benchmark")
    ahe_col, _ = _normalize_benchmark_columns(bench)
    _require_columns(bench, ["date", ahe_col], "benchmark")

    duplicates = bench.duplicated(subset=["date"], keep=False)
    if duplicates.any():
        bad = bench.loc[duplicates, ["date"]].drop_duplicates()
        raise ValueError(
            f"benchmark contains duplicate date rows: {bad.to_dict(orient='records')}"
        )

    required_dates = {base_date, end_date}
    if not required_dates.issubset(set(pd.to_datetime(bench["date"]).unique())):
        raise ValueError(
            "benchmark missing required endpoint dates: "
            f"expected {sorted(pd.to_datetime(list(required_dates)))}; got {sorted(pd.to_datetime(bench['date']).unique())}"
        )

    ahe_numeric = pd.to_numeric(bench[ahe_col], errors="coerce")
    if ahe_numeric.isna().any() or not np.isfinite(ahe_numeric.to_numpy()).all():
        raise ValueError(
            "benchmark contains missing, nonnumeric, or nonfinite published AHE values"
        )

    return bench, ahe_col


def _endpoint_merge(component_panel, weights, base_date, end_date):
    base_panel = component_panel[
        component_panel["date"] == pd.Timestamp(base_date)
    ].copy()
    end_panel = component_panel[
        component_panel["date"] == pd.Timestamp(end_date)
    ].copy()
    base_weights = weights[weights["date"] == pd.Timestamp(base_date)].copy()
    end_weights = weights[weights["date"] == pd.Timestamp(end_date)].copy()
    if "industry_name" not in base_weights.columns:
        base_weights["industry_name"] = pd.NA

    base_panel = base_panel[["industry_code", "ahe"]].rename(
        columns={"ahe": "base_ahe"}
    )
    end_panel = end_panel[["industry_code", "ahe"]].rename(columns={"ahe": "end_ahe"})
    base_weights = base_weights[["industry_code", "industry_name", "weight"]].rename(
        columns={"weight": "base_weight"}
    )
    end_weights = end_weights[["industry_code", "weight"]].rename(
        columns={"weight": "end_weight"}
    )

    if set(base_panel["industry_code"]) != set(base_weights["industry_code"]):
        raise ValueError(
            "missing component or weight rows for the base_date industry set"
        )
    if set(end_panel["industry_code"]) != set(end_weights["industry_code"]):
        raise ValueError(
            "missing component or weight rows for the end_date industry set"
        )

    base_merged = base_panel.merge(base_weights, on="industry_code", how="inner")
    end_merged = end_panel.merge(end_weights, on="industry_code", how="inner")

    if set(base_merged["industry_code"]) != set(end_merged["industry_code"]):
        raise ValueError(
            "component_panel and weights industry sets differ between base_date and end_date"
        )

    merged = base_merged.merge(end_merged, on="industry_code", how="inner")
    return merged.sort_values("industry_code").reset_index(drop=True)


def decompose_interval(component_panel, weights, benchmark, base_date, end_date):
    """Decompose a two-endpoint earnings change into within-industry and composition effects."""
    _require_frame(component_panel, "component_panel")
    _require_frame(weights, "weights")
    _require_frame(benchmark, "benchmark")

    base_ts = pd.Timestamp(base_date)
    end_ts = pd.Timestamp(end_date)
    if not (base_ts < end_ts):
        raise ValueError("base_date must be earlier than end_date")

    panel = _validate_component_frame(component_panel, base_ts, end_ts)
    weight_df = _validate_weights_frame(weights, base_ts, end_ts)
    benchmark_df, ahe_col = _validate_benchmark_frame(benchmark, base_ts, end_ts)

    if set(panel["date"].unique()) != {base_ts, end_ts}:
        raise ValueError("component_panel must contain exactly the base and end dates")
    if set(weight_df["date"].unique()) != {base_ts, end_ts}:
        raise ValueError("weights must contain exactly the base and end dates")
    if set(benchmark_df["date"].unique()) != {base_ts, end_ts}:
        raise ValueError("benchmark must contain exactly the base and end dates")

    merged = _endpoint_merge(panel, weight_df, base_ts, end_ts)

    merged["delta_ahe"] = merged["end_ahe"] - merged["base_ahe"]
    merged["delta_weight"] = merged["end_weight"] - merged["base_weight"]
    merged["within_contribution"] = (
        (merged["base_weight"] + merged["end_weight"]) / 2.0
    ) * merged["delta_ahe"]
    merged["composition_contribution"] = (
        (merged["base_ahe"] + merged["end_ahe"]) / 2.0
    ) * merged["delta_weight"]

    contributions = merged[
        [
            "industry_code",
            "industry_name",
            "base_ahe",
            "end_ahe",
            "base_weight",
            "end_weight",
            "within_contribution",
            "composition_contribution",
        ]
    ].copy()
    contributions["base_date"] = base_ts
    contributions["date"] = end_ts
    contributions = (
        contributions[
            [
                "base_date",
                "date",
                "industry_code",
                "industry_name",
                "base_ahe",
                "end_ahe",
                "base_weight",
                "end_weight",
                "within_contribution",
                "composition_contribution",
            ]
        ]
        .sort_values(["date", "industry_code"])
        .reset_index(drop=True)
    )

    base_lookup = panel[panel["date"] == base_ts][["industry_code", "ahe"]].merge(
        weight_df[weight_df["date"] == base_ts][["industry_code", "weight"]],
        on="industry_code",
        how="inner",
    )
    end_lookup = panel[panel["date"] == end_ts][["industry_code", "ahe"]].merge(
        weight_df[weight_df["date"] == end_ts][["industry_code", "weight"]],
        on="industry_code",
        how="inner",
    )

    reconstructed_base_ahe = float((base_lookup["ahe"] * base_lookup["weight"]).sum())
    reconstructed_end_ahe = float((end_lookup["ahe"] * end_lookup["weight"]).sum())
    reconstructed_change = reconstructed_end_ahe - reconstructed_base_ahe
    within_total = float(contributions["within_contribution"].sum())
    composition_total = float(contributions["composition_contribution"].sum())
    decomposition_residual = reconstructed_change - within_total - composition_total

    benchmark_by_date = benchmark_df.set_index("date")[ahe_col].to_dict()
    published_base_ahe = float(benchmark_by_date[base_ts])
    published_end_ahe = float(benchmark_by_date[end_ts])
    published_change = published_end_ahe - published_base_ahe
    benchmark_gap = published_end_ahe - reconstructed_end_ahe

    summary = pd.DataFrame(
        [
            {
                "base_date": base_ts,
                "date": end_ts,
                "reconstructed_base_ahe": reconstructed_base_ahe,
                "reconstructed_end_ahe": reconstructed_end_ahe,
                "reconstructed_change": reconstructed_change,
                "within_total": within_total,
                "composition_total": composition_total,
                "decomposition_residual": decomposition_residual,
                "published_base_ahe": published_base_ahe,
                "published_end_ahe": published_end_ahe,
                "published_change": published_change,
                "benchmark_gap": benchmark_gap,
            }
        ]
    )

    return contributions, summary


def _validate_sequence_inputs(component_panel, weights, benchmark, lag):
    if isinstance(lag, bool) or not isinstance(lag, int) or lag <= 0:
        raise ValueError("lag must be a positive integer")

    component = _coerce_date_series(component_panel, "component_panel")
    weight_df = _coerce_date_series(weights, "weights")
    bench = _coerce_date_series(benchmark, "benchmark")

    _require_columns(component, ["industry_code", "ahe"], "component_panel")
    _require_columns(weight_df, ["industry_code", "weight"], "weights")
    _require_columns(
        bench,
        ["date", "published_total_private_ahe", "published_total_private_employment"],
        "benchmark",
    )

    if component.empty or weight_df.empty or bench.empty:
        raise ValueError("component_panel, weights, and benchmark must not be empty")
    if not (component["date"].dt.day == 1).all():
        raise ValueError("component_panel dates must be first-of-month values")
    if not (weight_df["date"].dt.day == 1).all():
        raise ValueError("weights dates must be first-of-month values")
    if not (bench["date"].dt.day == 1).all():
        raise ValueError("benchmark dates must be first-of-month values")

    component_dates = set(pd.to_datetime(component["date"]).unique())
    weight_dates = set(pd.to_datetime(weight_df["date"]).unique())
    benchmark_dates = set(pd.to_datetime(bench["date"]).unique())

    if component_dates != weight_dates:
        raise ValueError("component_panel and weights have mismatched date coverage")
    if component_dates != benchmark_dates:
        raise ValueError(
            "benchmark date coverage does not match the selected component window"
        )

    all_dates = sorted(component_dates | weight_dates | benchmark_dates)
    expected_monthly_dates = pd.date_range(
        start=all_dates[0], end=all_dates[-1], freq="MS"
    )
    missing_months = sorted(set(expected_monthly_dates) - set(all_dates))
    if missing_months:
        raise ValueError(
            f"missing monthly dates in the decomposition sequence: {missing_months}"
        )

    component_industries_by_date = {
        date: set(group["industry_code"]) for date, group in component.groupby("date")
    }
    weight_industries_by_date = {
        date: set(group["industry_code"]) for date, group in weight_df.groupby("date")
    }

    reference_component_industries = next(iter(component_industries_by_date.values()))
    reference_weight_industries = next(iter(weight_industries_by_date.values()))
    if reference_component_industries != reference_weight_industries:
        raise ValueError("component_panel and weights have mismatched industry sets")
    for date, industries in component_industries_by_date.items():
        if industries != reference_component_industries:
            raise ValueError(
                f"component_panel has incomplete or mismatched industry coverage for date {date}"
            )
    for date, industries in weight_industries_by_date.items():
        if industries != reference_weight_industries:
            raise ValueError(
                f"weights have incomplete or mismatched industry coverage for date {date}"
            )
    if any(industry == "Total private" for industry in reference_component_industries):
        raise ValueError(
            "total-private benchmark rows must not appear as component industries"
        )

    if component.duplicated(subset=["date", "industry_code"]).any():
        raise ValueError("component_panel contains duplicate date/industry rows")
    if weight_df.duplicated(subset=["date", "industry_code"]).any():
        raise ValueError("weights contains duplicate date/industry rows")
    if bench.duplicated(subset=["date"]).any():
        raise ValueError("benchmark contains duplicate date rows")

    ahe_numeric = pd.to_numeric(component["ahe"], errors="coerce")
    if ahe_numeric.isna().any() or not np.isfinite(ahe_numeric.to_numpy()).all():
        raise ValueError(
            "component_panel contains missing, nonnumeric, or nonfinite ahe values"
        )

    weight_numeric = pd.to_numeric(weight_df["weight"], errors="coerce")
    if weight_numeric.isna().any() or not np.isfinite(weight_numeric.to_numpy()).all():
        raise ValueError("weights contains missing, nonnumeric, or nonfinite values")
    if (weight_numeric < 0).any():
        raise ValueError("weights must be nonnegative")

    by_date = weight_df.groupby("date")["weight"].sum()
    invalid_dates = by_date[~np.isclose(by_date, 1.0)].index.tolist()
    if invalid_dates:
        raise ValueError(
            "weights must sum to one for each date; invalid dates: "
            f"{sorted(pd.to_datetime(invalid_dates))}"
        )

    benchmark_ahe = pd.to_numeric(bench["published_total_private_ahe"], errors="coerce")
    if benchmark_ahe.isna().any() or not np.isfinite(benchmark_ahe.to_numpy()).all():
        raise ValueError(
            "benchmark contains missing, nonnumeric, or nonfinite published_total_private_ahe values"
        )

    return component, weight_df, bench, pd.DatetimeIndex(all_dates)


def build_decomposition_series(component_panel, weights, benchmark, lag=1):
    """Build a decomposition series from exact lagged month-over-month endpoint pairs."""
    if isinstance(lag, bool) or not isinstance(lag, int) or lag <= 0:
        raise ValueError("lag must be a positive integer")

    component = _coerce_date_series(component_panel, "component_panel")
    weight_df = _coerce_date_series(weights, "weights")
    bench = _coerce_date_series(benchmark, "benchmark")

    _require_columns(component, ["industry_code", "ahe"], "component_panel")
    _require_columns(weight_df, ["industry_code", "weight"], "weights")
    _require_columns(
        bench,
        ["date", "published_total_private_ahe", "published_total_private_employment"],
        "benchmark",
    )

    if component.empty or weight_df.empty or bench.empty:
        raise ValueError("component_panel, weights, and benchmark must not be empty")
    if component["date"].dt.day.ne(1).any():
        raise ValueError("component_panel dates must be first-of-month values")
    if weight_df["date"].dt.day.ne(1).any():
        raise ValueError("weights dates must be first-of-month values")
    if bench["date"].dt.day.ne(1).any():
        raise ValueError("benchmark dates must be first-of-month values")

    if component.duplicated(subset=["date", "industry_code"]).any():
        raise ValueError("component_panel contains duplicate date/industry rows")
    if weight_df.duplicated(subset=["date", "industry_code"]).any():
        raise ValueError("weights contains duplicate date/industry rows")
    if bench.duplicated(subset=["date"]).any():
        raise ValueError("benchmark contains duplicate date rows")

    component_dates = sorted(pd.to_datetime(component["date"]).unique())
    weight_dates = sorted(pd.to_datetime(weight_df["date"]).unique())
    benchmark_dates = sorted(pd.to_datetime(bench["date"]).unique())
    if component_dates != weight_dates:
        raise ValueError("component_panel and weights have mismatched date coverage")
    if set(component_dates) != set(benchmark_dates):
        raise ValueError(
            "benchmark date coverage does not match the selected component window"
        )

    for date in component_dates:
        date_component = component[component["date"] == date]
        date_weights = weight_df[weight_df["date"] == date]
        if set(date_component["industry_code"]) != set(date_weights["industry_code"]):
            raise ValueError(
                f"component_panel and weights have mismatched industry coverage for date {date}"
            )

    ahe_numeric = pd.to_numeric(component["ahe"], errors="coerce")
    if ahe_numeric.isna().any() or not np.isfinite(ahe_numeric.to_numpy()).all():
        raise ValueError(
            "component_panel contains missing, nonnumeric, or nonfinite ahe values"
        )
    weight_numeric = pd.to_numeric(weight_df["weight"], errors="coerce")
    if weight_numeric.isna().any() or not np.isfinite(weight_numeric.to_numpy()).all():
        raise ValueError("weights contains missing, nonnumeric, or nonfinite values")
    if (weight_numeric < 0).any():
        raise ValueError("weights must be nonnegative")
    if not np.allclose(weight_df.groupby("date")["weight"].sum().to_numpy(), 1.0):
        raise ValueError("weights must sum to one for each date")

    benchmark_ahe = pd.to_numeric(bench["published_total_private_ahe"], errors="coerce")
    if benchmark_ahe.isna().any() or not np.isfinite(benchmark_ahe.to_numpy()).all():
        raise ValueError(
            "benchmark contains missing, nonnumeric, or nonfinite published_total_private_ahe values"
        )

    end_dates = pd.DatetimeIndex(
        sorted(set(component_dates) | set(weight_dates) | set(benchmark_dates))
    )
    if len(end_dates) == 0:
        raise ValueError("no dates available for decomposition")

    expected_dates = pd.date_range(start=end_dates[0], end=end_dates[-1], freq="MS")
    if not set(expected_dates) == set(end_dates):
        missing = sorted(set(expected_dates) - set(end_dates))
        raise ValueError(
            f"missing monthly dates in the decomposition sequence: {missing}"
        )

    interval_contributions = []
    interval_summaries = []

    for idx in range(lag, len(end_dates)):
        end_date = end_dates[idx]
        base_date = end_date - pd.DateOffset(months=lag)
        if base_date not in end_dates:
            raise ValueError(
                f"missing exact base month {base_date} for end date {end_date} with lag={lag}"
            )

        subset_component = component[
            component["date"].isin([base_date, end_date])
        ].copy()
        subset_weights = weight_df[weight_df["date"].isin([base_date, end_date])].copy()
        subset_benchmark = bench[bench["date"].isin([base_date, end_date])].copy()

        if set(subset_component["date"].unique()) != {base_date, end_date}:
            raise ValueError(
                f"component_panel is missing an endpoint month in the {base_date} -> {end_date} interval"
            )
        if set(subset_weights["date"].unique()) != {base_date, end_date}:
            raise ValueError(
                f"weights are missing an endpoint month in the {base_date} -> {end_date} interval"
            )
        if set(subset_benchmark["date"].unique()) != {base_date, end_date}:
            raise ValueError(
                f"benchmark is missing an endpoint month in the {base_date} -> {end_date} interval"
            )

        period_contributions, period_summary = decompose_interval(
            subset_component,
            subset_weights,
            subset_benchmark,
            base_date,
            end_date,
        )
        period_contributions = period_contributions.copy()
        period_summary = period_summary.copy()
        period_contributions["lag"] = lag
        period_summary["lag"] = lag
        period_summary["comparison_label"] = (
            "trailing-12-month endpoint comparison"
            if lag == 12
            else "monthly endpoint comparison"
        )
        interval_contributions.append(period_contributions)
        interval_summaries.append(period_summary)

    if not interval_contributions:
        raise ValueError(
            "no valid decomposition intervals were available for the selected lag"
        )

    contributions = pd.concat(interval_contributions, ignore_index=True)
    summary = pd.concat(interval_summaries, ignore_index=True)
    contributions = contributions.sort_values(
        ["base_date", "date", "industry_code"]
    ).reset_index(drop=True)
    summary = summary.sort_values(["base_date", "date"]).reset_index(drop=True)

    return contributions, summary


def rank_decomposition_contributions(contributions, date, effect="composition"):
    """Return a signed contribution table ordered by absolute contribution magnitude."""
    _require_frame(contributions, "contributions")
    _require_columns(contributions, ["date", "industry_code"], "contributions")

    if effect not in {"within", "composition"}:
        raise ValueError(f"effect must be 'within' or 'composition'; got {effect!r}")

    effect_column = f"{effect}_contribution"
    if effect_column not in contributions.columns:
        raise ValueError(f"contributions table does not contain {effect_column!r}")

    target_date = pd.Timestamp(date)
    selected = contributions[contributions["date"] == target_date].copy()
    if selected.empty:
        raise ValueError(
            f"date {target_date} is not present in the contributions table"
        )

    selected = selected.copy()
    selected["abs_effect"] = selected[effect_column].abs()
    selected = (
        selected.sort_values(
            ["abs_effect", "industry_code"],
            ascending=[False, True],
            kind="mergesort",
        )
        .drop(columns=["abs_effect"])
        .reset_index(drop=True)
    )
    return selected


def build_fixed_composition_series(
    component_panel,
    weights,
    benchmark,
    reference_date,
):
    """Compute a fixed-composition counterfactual using the reference-date weights."""
    component = _coerce_date_series(component_panel, "component_panel")
    _require_columns(component, ["industry_code", "ahe"], "component_panel")

    weight_df = _coerce_date_series(weights, "weights")
    _require_columns(weight_df, ["industry_code", "weight"], "weights")

    benchmark_df = _coerce_date_series(benchmark, "benchmark")
    _require_columns(
        benchmark_df,
        ["date", "published_total_private_ahe", "published_total_private_employment"],
        "benchmark",
    )

    if component.duplicated(subset=["date", "industry_code"]).any():
        raise ValueError(
            "component_panel contains duplicate (date, industry_code) rows"
        )
    if weight_df.duplicated(subset=["date", "industry_code"]).any():
        raise ValueError("weights contains duplicate (date, industry_code) rows")
    if benchmark_df.duplicated(subset=["date"]).any():
        raise ValueError("benchmark contains duplicate date rows")

    ahe_numeric = pd.to_numeric(component["ahe"], errors="coerce")
    if ahe_numeric.isna().any() or not np.isfinite(ahe_numeric.to_numpy()).all():
        raise ValueError(
            "component_panel contains missing, nonnumeric, or nonfinite ahe values"
        )

    weight_numeric = pd.to_numeric(weight_df["weight"], errors="coerce")
    if weight_numeric.isna().any() or not np.isfinite(weight_numeric.to_numpy()).all():
        raise ValueError("weights contains missing, nonnumeric, or nonfinite values")
    if (weight_numeric < 0).any():
        raise ValueError("weights must be nonnegative")

    pub_ahe_numeric = pd.to_numeric(
        benchmark_df["published_total_private_ahe"], errors="coerce"
    )
    if (
        pub_ahe_numeric.isna().any()
        or not np.isfinite(pub_ahe_numeric.to_numpy()).all()
    ):
        raise ValueError(
            "benchmark contains missing, nonnumeric, or nonfinite published_total_private_ahe values"
        )

    component_dates = sorted(pd.to_datetime(component["date"].unique()))
    weight_dates = sorted(pd.to_datetime(weight_df["date"].unique()))
    benchmark_dates = sorted(pd.to_datetime(benchmark_df["date"].unique()))
    if component_dates != weight_dates:
        raise ValueError("component_panel and weights have mismatched date coverage")
    if sorted(set(component_dates) - set(benchmark_dates)):
        raise ValueError(
            "benchmark is missing required dates for the selected component window"
        )

    ref_ts = pd.Timestamp(reference_date)
    if pd.isna(ref_ts) or ref_ts not in set(component_dates):
        raise ValueError(
            "reference_date is outside the available component and weight window"
        )

    for date in component_dates:
        date_component = component[component["date"] == date]
        date_weights = weight_df[weight_df["date"] == date]
        if set(date_component["industry_code"]) != set(date_weights["industry_code"]):
            raise ValueError(
                f"component_panel and weights have mismatched industry coverage for date {date}"
            )
    date_weight_sums = weight_df.groupby("date")["weight"].sum()
    invalid_weight_dates = date_weight_sums[
        ~np.isclose(date_weight_sums, 1.0)
    ].index.tolist()
    if invalid_weight_dates:
        raise ValueError(
            "weights must sum to one for each date; invalid dates: "
            f"{sorted(pd.to_datetime(invalid_weight_dates))}"
        )

    reference_weights = weight_df[weight_df["date"] == ref_ts].copy()
    if reference_weights.empty:
        raise ValueError(f"reference_date weight row is missing for {ref_ts}")
    if reference_weights["industry_code"].duplicated().any():
        raise ValueError(f"reference_date weight rows are duplicated for {ref_ts}")

    reference_component = component[component["date"] == ref_ts].copy()
    if set(reference_component["industry_code"]) != set(
        reference_weights["industry_code"]
    ):
        raise ValueError(
            f"reference-date component and weight industry sets are misaligned for {ref_ts}"
        )

    reference_weight_map = reference_weights.set_index("industry_code")["weight"]
    if not np.isclose(reference_weight_map.sum(), 1.0):
        raise ValueError(f"reference-date weights must sum to one for {ref_ts}")

    reference_component_by_industry = reference_component.set_index("industry_code")[
        "ahe"
    ]
    fixed_reference_ahe = float(
        (
            reference_component_by_industry
            * reference_weight_map.reindex(reference_component_by_industry.index)
        ).sum()
    )
    if fixed_reference_ahe <= 0:
        raise ValueError(
            "reference_date fixed-composition value must be positive to compute growth"
        )

    rows = []
    for date in component_dates:
        date_component = component[component["date"] == date].set_index(
            "industry_code"
        )["ahe"]
        date_weights = weight_df[weight_df["date"] == date].set_index("industry_code")[
            "weight"
        ]
        if set(date_component.index) != set(date_weights.index):
            raise ValueError(
                f"component_panel and weights have mismatched industry coverage for date {date}"
            )

        reconstructed_ahe = float(
            (date_component * date_weights.reindex(date_component.index)).sum()
        )
        fixed_composition_ahe = float(
            (date_component * reference_weight_map.reindex(date_component.index)).sum()
        )

        published = float(
            benchmark_df.loc[
                benchmark_df["date"] == pd.Timestamp(date),
                "published_total_private_ahe",
            ].iat[0]
        )
        fixed_composition_growth_pct = 100.0 * (
            fixed_composition_ahe / fixed_reference_ahe - 1.0
        )
        reconstructed_growth_pct = 100.0 * (
            reconstructed_ahe / fixed_reference_ahe - 1.0
        )
        benchmark_gap = published - reconstructed_ahe

        rows.append(
            {
                "date": pd.Timestamp(date),
                "reference_date": ref_ts,
                "fixed_composition_ahe": fixed_composition_ahe,
                "reconstructed_ahe": reconstructed_ahe,
                "published_total_private_ahe": published,
                "fixed_composition_growth_pct": fixed_composition_growth_pct,
                "reconstructed_growth_pct": reconstructed_growth_pct,
                "benchmark_gap": benchmark_gap,
            }
        )

    result = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return result


__all__ = [
    "decompose_interval",
    "build_decomposition_series",
    "build_fixed_composition_series",
    "rank_decomposition_contributions",
]
