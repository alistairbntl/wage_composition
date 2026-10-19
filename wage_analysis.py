from datetime import date, datetime, timezone
import json
import math
from pathlib import Path

import pandas as pd
import requests

"""Stage 1A CES series specification.

Scope is monthly, seasonally adjusted, all-employees CES measures in source
units. Total private is included as a benchmark (not a component).
"""

_INDUSTRIES = [
    ("10000000", "Mining and logging", "component"),
    ("20000000", "Construction", "component"),
    ("30000000", "Manufacturing", "component"),
    ("40000000", "Trade, transportation, and utilities", "component"),
    ("50000000", "Information", "component"),
    ("55000000", "Financial activities", "component"),
    ("60000000", "Professional and business services", "component"),
    ("65000000", "Private education and health services", "component"),
    ("70000000", "Leisure and hospitality", "component"),
    ("80000000", "Other services", "component"),
    ("05000000", "Total private", "benchmark"),
]

_METRICS = [
    ("employment", "01", "thousand_employees"),
    ("average_weekly_hours", "02", "hours_per_week"),
    ("average_hourly_earnings", "03", "nominal_usd_per_hour"),
]

_SOURCE_URL = "https://www.bls.gov/web/empsit/cesseriespub.htm"

SERIES_SPEC = [
    {
        "series_id": f"CES{industry_code}{suffix}",
        "industry_code": industry_code,
        "industry_name": industry_name,
        "metric": metric,
        "unit": unit,
        "frequency": "monthly",
        "seasonal_adjustment": "seasonally_adjusted",
        "employee_population": "all_employees",
        "role": role,
        "source_url": _SOURCE_URL,
    }
    for industry_code, industry_name, role in _INDUSTRIES
    for metric, suffix, unit in _METRICS
]


def build_request_payloads(start_year=2019, end_year=None, *, as_of_year=None):
    effective_end_year = (
        end_year
        if end_year is not None
        else as_of_year
        if as_of_year is not None
        else date.today().year
    )

    if effective_end_year < start_year:
        raise ValueError(
            f"Invalid year range: start_year {start_year} is after end_year {effective_end_year}."
        )

    requested_years = effective_end_year - start_year + 1
    if requested_years > 10:
        raise ValueError(
            "BLS v1 requests support at most 10 inclusive years; "
            f"got range {start_year}-{effective_end_year} ({requested_years} years)."
        )

    series_ids = [record["series_id"] for record in SERIES_SPEC]
    batch_size = 25
    payloads = []

    for offset in range(0, len(series_ids), batch_size):
        batch_ids = series_ids[offset : offset + batch_size]
        payloads.append(
            {
                "seriesid": batch_ids,
                "startyear": str(start_year),
                "endyear": str(effective_end_year),
            }
        )

    return payloads


class BLSRequestError(RuntimeError):
    """Raised when a BLS batch request fails during transport."""


_BLS_V1_ENDPOINT = "https://api.bls.gov/publicAPI/v1/timeseries/data/"


def fetch_bls_batch(payload, *, timeout=30):
    if timeout <= 0 or not math.isfinite(timeout):
        raise ValueError(f"timeout must be a finite positive number, got {timeout}")

    start_year = payload.get("startyear", "?")
    end_year = payload.get("endyear", "?")
    series_ids = payload.get("seriesid") or []
    batch_context = f"years {start_year}-{end_year}, series_count={len(series_ids)}"

    try:
        response = requests.post(_BLS_V1_ENDPOINT, json=payload, timeout=timeout)
    except (requests.Timeout, requests.ConnectionError) as exc:
        raise BLSRequestError(
            f"BLS request transport failed for batch ({batch_context})"
        ) from exc

    if not 200 <= response.status_code < 300:
        raise BLSRequestError(
            f"BLS request failed with HTTP {response.status_code} for batch ({batch_context})"
        )

    try:
        envelope = response.json()
    except Exception as exc:
        raise BLSRequestError(
            f"BLS response JSON decode failed for batch ({batch_context})"
        ) from exc

    if not isinstance(envelope, dict):
        raise BLSRequestError(
            f"Malformed BLS response envelope for batch ({batch_context}): expected mapping"
        )

    if "status" not in envelope or "Results" not in envelope:
        raise BLSRequestError(
            f"Malformed BLS response envelope for batch ({batch_context}): missing required fields"
        )

    status = envelope["status"]
    message = envelope.get("message", [])
    results = envelope["Results"]

    if not isinstance(status, str):
        raise BLSRequestError(
            f"Malformed BLS response envelope for batch ({batch_context}): status must be a string"
        )
    if "message" in envelope and not isinstance(message, list):
        raise BLSRequestError(
            f"Malformed BLS response envelope for batch ({batch_context}): message must be a list"
        )
    valid_results = isinstance(results, list) or (
        isinstance(results, dict)
        and "series" in results
        and isinstance(results["series"], list)
    )
    if not valid_results:
        raise BLSRequestError(
            f"Malformed BLS response envelope for batch ({batch_context}): Results must be a list or a mapping with series list"
        )

    if status != "REQUEST_SUCCEEDED":
        raise BLSRequestError(
            "BLS application failure for batch "
            f"({batch_context}): status={status}, message={message}"
        )

    return envelope


def parse_bls_response(envelope, requested_series_ids, start_year, end_year):
    results = envelope.get("Results", [])
    if isinstance(results, dict):
        series_blocks = results.get("series", [])
    else:
        series_blocks = results

    requested_ids = set(requested_series_ids)
    returned_ids = {
        block.get("seriesID") or block.get("seriesId") for block in series_blocks
    }

    missing_ids = sorted(requested_ids - returned_ids)
    if missing_ids:
        raise BLSRequestError(f"Requested series missing from response: {missing_ids}")

    unexpected_ids = sorted(returned_ids - requested_ids)
    if unexpected_ids:
        raise BLSRequestError(
            f"Unexpected series returned in response: {unexpected_ids}"
        )

    rows = []
    seen_keys = set()
    monthly_counts = {series_id: 0 for series_id in requested_ids}

    for series_block in series_blocks:
        series_id = series_block.get("seriesID") or series_block.get("seriesId")
        for obs in series_block.get("data", []):
            period = obs.get("period")
            year = obs.get("year")

            if period == "M13":
                continue

            if not isinstance(period, str) or not period.startswith("M"):
                raise ValueError(
                    f"Unexpected period code {period!r} for series {series_id} in year {year}"
                )

            month = period[1:]
            if month not in {f"{m:02d}" for m in range(1, 13)}:
                raise ValueError(
                    f"Unexpected period code {period!r} for series {series_id} in year {year}"
                )

            try:
                year_int = int(year)
            except Exception as exc:
                raise BLSRequestError(
                    f"Invalid observation year {year!r} for series {series_id}, period {period}"
                ) from exc

            if year_int < start_year or year_int > end_year:
                raise BLSRequestError(
                    "Observation year out of requested range: "
                    f"series={series_id}, year={year_int}, period={period}, "
                    f"range={start_year}-{end_year}"
                )

            raw_value = obs.get("value")
            try:
                value = float(raw_value)
            except Exception as exc:
                raise BLSRequestError(
                    f"Non-numeric observation value {raw_value!r} for series {series_id}, year {year_int}, period {period}"
                ) from exc

            if not math.isfinite(value):
                raise BLSRequestError(
                    f"Non-finite observation value {raw_value!r} for series {series_id}, year {year_int}, period {period}"
                )

            obs_date = pd.Timestamp(year_int, int(month), 1)
            key = (series_id, obs_date)
            if key in seen_keys:
                raise BLSRequestError(
                    f"Duplicate observation for series {series_id} on date {obs_date.date()}"
                )
            seen_keys.add(key)

            footnotes = []
            raw_footnotes = obs.get("footnotes")
            if isinstance(raw_footnotes, list):
                for note in raw_footnotes:
                    if not isinstance(note, dict):
                        continue
                    code = note.get("code", "")
                    text = note.get("text", "")
                    if code == "" and text == "":
                        continue
                    footnotes.append(note)

            rows.append(
                {
                    "series_id": series_id,
                    "date": obs_date,
                    "value": value,
                    "footnotes": footnotes,
                }
            )
            monthly_counts[series_id] += 1

    empty_series = sorted(
        series_id for series_id, count in monthly_counts.items() if count == 0
    )
    if empty_series:
        raise BLSRequestError(
            f"Requested series with no monthly observations: {empty_series}"
        )

    table = pd.DataFrame(rows, columns=["series_id", "date", "value", "footnotes"])
    table = table.sort_values(["series_id", "date"]).reset_index(drop=True)
    return table


_SNAPSHOT_FORMAT_VERSION = 1


def _snapshot_filename(now_utc):
    timestamp = now_utc.strftime("%Y%m%dT%H%M%S%fZ")
    return f"snapshot_{timestamp}.json"


def candidate_common_window(report):
    """Return the candidate intersection over all required series in a coverage report."""
    if report is None or len(report) == 0:
        raise ValueError(
            "coverage report is empty; cannot determine a candidate window"
        )

    series_first = pd.to_datetime(report["first_date"], errors="coerce").dropna()
    series_last = pd.to_datetime(report["last_date"], errors="coerce").dropna()

    if series_first.empty or series_last.empty:
        raise ValueError("coverage report is missing first/last date information")

    start = series_first.max().strftime("%Y-%m-%d")
    end = series_last.min().strftime("%Y-%m-%d")
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError(
            "No common overlap across required series: "
            f"latest_start={start}, earliest_end={end}"
        )
    return start, end


def validate_selected_window(report, start_date, end_date):
    """Validate an explicitly chosen window against the coverage report."""
    if report is None or len(report) == 0:
        raise ValueError("coverage report is empty; cannot validate a selected window")

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    if start_ts > end_ts:
        raise ValueError(
            f"Selected window start_date {start_date!r} is after end_date {end_date!r}"
        )

    try:
        candidate_start, candidate_end = candidate_common_window(report)
    except ValueError as exc:
        raise ValueError(f"Invalid candidate window: {exc}") from exc

    candidate_start_ts = pd.Timestamp(candidate_start).normalize()
    candidate_end_ts = pd.Timestamp(candidate_end).normalize()

    if start_ts < candidate_start_ts or end_ts > candidate_end_ts:
        raise ValueError(
            "Selected window is outside the observed common bounds: "
            f"selected={start_date} to {end_date}, candidate={candidate_start} to {candidate_end}"
        )

    selected_months = list(pd.date_range(start=start_ts, end=end_ts, freq="MS"))
    selected_count = len(selected_months)

    for row in report.itertuples(index=False):
        series_id = getattr(row, "series_id", None)
        status = getattr(row, "status", "")
        missing_months = getattr(row, "missing_months", None) or []
        first_date = pd.to_datetime(getattr(row, "first_date", None), errors="coerce")
        last_date = pd.to_datetime(getattr(row, "last_date", None), errors="coerce")

        if pd.notna(first_date) and first_date > start_ts:
            raise ValueError(
                f"Selected window {start_date} to {end_date} starts before series {series_id} is observed: "
                f"first_date={first_date.strftime('%Y-%m-%d')}"
            )
        if pd.notna(last_date) and last_date < end_ts:
            raise ValueError(
                f"Selected window {start_date} to {end_date} ends after series {series_id} is observed: "
                f"last_date={last_date.strftime('%Y-%m-%d')}"
            )

        if missing_months:
            missing_in_window = [
                month
                for month in missing_months
                if pd.Timestamp(month).normalize() >= start_ts
                and pd.Timestamp(month).normalize() <= end_ts
            ]
            if missing_in_window:
                return False

        observed_count = getattr(row, "observed_count", None)
        expected_count = getattr(row, "expected_count", None)
        if observed_count is not None and expected_count is not None:
            if observed_count != expected_count:
                return False

        if status != "complete":
            return False

        if observed_count is not None and observed_count != selected_count:
            return False

    return True


def audit_coverage(observations, series_spec, start_date, end_date):
    """Audit a snapshot against the required monthly series coverage.

    The function builds a per-series coverage report from the explicit
    `series_spec` contract rather than from whichever IDs happen to appear in the
    observations. It enforces an explicit monthly window and reports any missing
    first-of-month dates without filling, dropping, or reindexing data.
    """
    if observations is None:
        raise ValueError("observations cannot be None")
    if series_spec is None:
        raise ValueError("series_spec cannot be None")

    obs = observations.copy()
    if not isinstance(obs, pd.DataFrame):
        raise TypeError("observations must be a pandas DataFrame")
    if "series_id" not in obs.columns or "date" not in obs.columns:
        raise ValueError("observations must include 'series_id' and 'date' columns")

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    if start_ts > end_ts:
        raise ValueError(
            f"start_date {start_date!r} must be on or before end_date {end_date!r}"
        )

    expected_dates = pd.date_range(start=start_ts, end=end_ts, freq="MS")
    expected_months = {date_value.strftime("%Y-%m-%d") for date_value in expected_dates}
    expected_count = len(expected_dates)
    expected_start_date = expected_dates[0].strftime("%Y-%m-%d")
    expected_end_date = expected_dates[-1].strftime("%Y-%m-%d")

    obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()
    series_records = list(series_spec)
    report_rows = []

    for record in series_records:
        series_id = record["series_id"]
        relevant = obs[obs["series_id"] == series_id].copy()
        relevant = relevant[relevant["date"].between(start_ts, end_ts)]

        raw_dates = pd.DatetimeIndex(relevant["date"]).sort_values()
        unique_dates = pd.DatetimeIndex(raw_dates.unique())
        observed_count = int(len(relevant))
        first_date = unique_dates[0] if len(unique_dates) else pd.NaT
        last_date = unique_dates[-1] if len(unique_dates) else pd.NaT
        observed_start_date = (
            first_date.strftime("%Y-%m-%d") if pd.notna(first_date) else None
        )
        observed_end_date = (
            last_date.strftime("%Y-%m-%d") if pd.notna(last_date) else None
        )

        observed_months = {d.strftime("%Y-%m-%d") for d in unique_dates}
        missing_months = sorted(expected_months - observed_months)
        duplicate_dates = relevant["date"].duplicated(keep=False).any()

        if (
            observed_count == expected_count
            and not missing_months
            and not duplicate_dates
        ):
            status = "complete"
        elif duplicate_dates:
            status = "duplicate_observations"
        elif missing_months:
            status = "missing_observations"
        elif observed_count == 0:
            status = "no_observations"
        else:
            status = "partial"

        report_rows.append(
            {
                "series_id": series_id,
                "industry_code": record.get("industry_code"),
                "industry_name": record.get("industry_name"),
                "metric": record.get("metric"),
                "unit": record.get("unit"),
                "role": record.get("role"),
                "first_date": first_date,
                "last_date": last_date,
                "observed_start_date": observed_start_date,
                "observed_end_date": observed_end_date,
                "expected_start_date": expected_start_date,
                "expected_end_date": expected_end_date,
                "observed_count": observed_count,
                "expected_count": expected_count,
                "missing_months": missing_months,
                "status": status,
            }
        )

    report = pd.DataFrame(
        report_rows,
        columns=[
            "series_id",
            "industry_code",
            "industry_name",
            "metric",
            "unit",
            "role",
            "first_date",
            "last_date",
            "observed_start_date",
            "observed_end_date",
            "expected_start_date",
            "expected_end_date",
            "observed_count",
            "expected_count",
            "missing_months",
            "status",
        ],
    )
    return report.reset_index(drop=True)


def build_analysis_panel(observations, series_spec, start_date, end_date):
    """Build a tidy component panel for the selected monthly window."""
    if observations is None:
        raise ValueError("observations cannot be None")
    if series_spec is None:
        raise ValueError("series_spec cannot be None")
    if not isinstance(observations, pd.DataFrame):
        raise TypeError("observations must be a pandas DataFrame")

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    if start_ts > end_ts:
        raise ValueError(
            f"start_date {start_date!r} must be on or before end_date {end_date!r}"
        )

    obs = observations.copy()
    obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()

    component_records = [
        record for record in series_spec if record.get("role") == "component"
    ]
    series_lookup = {record["series_id"]: record for record in component_records}
    if not series_lookup:
        return pd.DataFrame(
            columns=[
                "date",
                "industry_code",
                "industry_name",
                "role",
                "employment",
                "average_weekly_hours",
                "average_hourly_earnings",
                "employment_unit",
                "average_weekly_hours_unit",
                "average_hourly_earnings_unit",
            ]
        )

    selected = obs[obs["date"].between(start_ts, end_ts, inclusive="both")].copy()
    out_of_window = obs[~obs["date"].between(start_ts, end_ts, inclusive="both")]
    if not out_of_window.empty:
        offending = out_of_window.iloc[0]
        raise ValueError(
            "Observation outside selected window: "
            f"series_id={offending.get('series_id')}, date={offending.get('date').strftime('%Y-%m-%d')}, "
            f"value={offending.get('value')}"
        )

    if selected.empty:
        raise ValueError(
            f"No component observations available for selected window {start_ts.date()} to {end_ts.date()}"
        )

    selected = selected[selected["series_id"].isin(series_lookup)].copy()
    if selected.empty:
        raise ValueError(
            f"No component observations available for selected window {start_ts.date()} to {end_ts.date()}"
        )

    for row in selected.itertuples(index=False):
        series_id = getattr(row, "series_id")
        value = getattr(row, "value")
        date_value = pd.Timestamp(getattr(row, "date")).normalize()
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Non-numeric value for series {series_id} on {date_value.strftime('%Y-%m-%d')}: {value!r}"
            )
        if not math.isfinite(numeric_value):
            raise ValueError(
                f"Non-finite value for series {series_id} on {date_value.strftime('%Y-%m-%d')}: {value!r}"
            )

    duplicates = selected[selected.duplicated(subset=["series_id", "date"], keep=False)]
    if not duplicates.empty:
        problem = duplicates.iloc[0]
        raise ValueError(
            "duplicate observation in selected window: "
            f"series_id={problem['series_id']}, date={problem['date'].strftime('%Y-%m-%d')}"
        )

    industry_to_metrics = {}
    for record in component_records:
        industry_to_metrics.setdefault(record["industry_code"], set()).add(
            record["metric"]
        )

    available_industries = sorted(
        {
            series_lookup[series_id]["industry_code"]
            for series_id in selected["series_id"].unique()
        }
    )
    months = list(pd.date_range(start=start_ts, end=end_ts, freq="MS"))
    errors = []

    for industry_code in available_industries:
        expected_metrics = sorted(industry_to_metrics.get(industry_code, set()))
        for month in months:
            month_rows = selected[
                (selected["date"] == month)
                & (
                    selected["series_id"].isin(
                        [
                            series_id
                            for series_id, record in series_lookup.items()
                            if record["industry_code"] == industry_code
                        ]
                    )
                )
            ]
            observed_metrics = {
                series_lookup[series_id]["metric"]
                for series_id in month_rows["series_id"].unique()
                if series_id in series_lookup
            }
            missing_metrics = sorted(set(expected_metrics) - observed_metrics)
            if missing_metrics:
                errors.append(
                    f"missing metric for industry {industry_code} on {month.strftime('%Y-%m-%d')}: {missing_metrics[0]}"
                )
            if len(month_rows) != len(expected_metrics):
                if len(month_rows) < len(expected_metrics):
                    errors.append(
                        f"missing observations for industry {industry_code} on {month.strftime('%Y-%m-%d')}"
                    )
                elif len(month_rows) > len(expected_metrics):
                    errors.append(
                        f"duplicate metric observations for industry {industry_code} on {month.strftime('%Y-%m-%d')}"
                    )
    if errors:
        raise ValueError(errors[0])

    panel_rows = []
    for month in months:
        for industry_code in available_industries:
            industry_record = next(
                record
                for record in component_records
                if record["industry_code"] == industry_code
            )
            row_payload = {
                "date": month,
                "industry_code": industry_code,
                "industry_name": industry_record["industry_name"],
                "role": "component",
            }
            for metric in [
                "employment",
                "average_weekly_hours",
                "average_hourly_earnings",
            ]:
                metric_rows = selected[
                    (selected["date"] == month)
                    & (
                        selected["series_id"].isin(
                            [
                                series_id
                                for series_id, record in series_lookup.items()
                                if record["industry_code"] == industry_code
                                and record["metric"] == metric
                            ]
                        )
                    )
                ]
                if len(metric_rows) != 1:
                    raise ValueError(
                        f"missing metric for industry {industry_code} on {month.strftime('%Y-%m-%d')}: {metric}"
                    )
                row_payload[metric] = float(metric_rows.iloc[0]["value"])
            row_payload["employment_unit"] = next(
                record["unit"]
                for record in component_records
                if record["industry_code"] == industry_code
                and record["metric"] == "employment"
            )
            row_payload["average_weekly_hours_unit"] = next(
                record["unit"]
                for record in component_records
                if record["industry_code"] == industry_code
                and record["metric"] == "average_weekly_hours"
            )
            row_payload["average_hourly_earnings_unit"] = next(
                record["unit"]
                for record in component_records
                if record["industry_code"] == industry_code
                and record["metric"] == "average_hourly_earnings"
            )
            panel_rows.append(row_payload)

    panel = pd.DataFrame(
        panel_rows,
        columns=[
            "date",
            "industry_code",
            "industry_name",
            "role",
            "employment",
            "average_weekly_hours",
            "average_hourly_earnings",
            "employment_unit",
            "average_weekly_hours_unit",
            "average_hourly_earnings_unit",
        ],
    )
    panel = panel.sort_values(["date", "industry_code"]).reset_index(drop=True)
    return panel


def calculate_paid_hours_weights(component_panel):
    """Calculate monthly paid-hours exposure and component weights."""
    if component_panel is None:
        raise ValueError("component_panel cannot be None")
    if not isinstance(component_panel, pd.DataFrame):
        raise TypeError("component_panel must be a pandas DataFrame")

    required_cols = {
        "date",
        "industry_code",
        "employment",
        "average_weekly_hours",
    }
    missing_cols = sorted(required_cols - set(component_panel.columns))
    if missing_cols:
        raise ValueError(f"component_panel is missing required columns: {missing_cols}")

    if "role" in component_panel.columns:
        benchmark_rows = component_panel[component_panel["role"].eq("benchmark")]
        if not benchmark_rows.empty:
            benchmark = benchmark_rows.iloc[0]
            raise ValueError(
                "benchmark rows are not valid component weights: "
                f"industry_code={benchmark.get('industry_code')}, "
                f"date={pd.Timestamp(benchmark.get('date')).strftime('%Y-%m-%d')}, role=benchmark"
            )

    benchmark_code_rows = component_panel[
        component_panel["industry_code"].astype(str).str.startswith("05000000")
    ]
    if not benchmark_code_rows.empty:
        benchmark = benchmark_code_rows.iloc[0]
        raise ValueError(
            "benchmark series must be excluded from component weights: "
            f"industry_code={benchmark.get('industry_code')}, "
            f"date={pd.Timestamp(benchmark.get('date')).strftime('%Y-%m-%d')}"
        )

    panel = component_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()

    for col in ["employment", "average_weekly_hours"]:
        try:
            panel[col] = panel[col].astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric {col} in component panel") from exc

    bad_values = panel[
        ~panel["employment"].apply(math.isfinite)
        | ~panel["average_weekly_hours"].apply(math.isfinite)
    ]
    if not bad_values.empty:
        bad = bad_values.iloc[0]
        raise ValueError(
            "Non-finite employment or hours for component panel: "
            f"industry_code={bad.get('industry_code')}, date={pd.Timestamp(bad.get('date')).strftime('%Y-%m-%d')}, "
            f"employment={bad.get('employment')}, average_weekly_hours={bad.get('average_weekly_hours')}"
        )

    negative_income = panel[
        (panel["employment"] < 0) | (panel["average_weekly_hours"] < 0)
    ]
    if not negative_income.empty:
        bad = negative_income.iloc[0]
        raise ValueError(
            "negative employment or average_weekly_hours is invalid: "
            f"industry_code={bad.get('industry_code')}, date={pd.Timestamp(bad.get('date')).strftime('%Y-%m-%d')}"
        )

    panel["paid_hours"] = panel["employment"] * panel["average_weekly_hours"]
    monthly_totals = (
        panel.groupby("date", as_index=False)["paid_hours"]
        .sum()
        .rename(columns={"paid_hours": "total_paid_hours"})
    )
    panel = panel.merge(monthly_totals, on="date", how="left")
    zero_total = panel[panel["total_paid_hours"] <= 0]
    if not zero_total.empty:
        bad = zero_total.iloc[0]
        raise ValueError(
            "total paid-hours denominator is zero or invalid for component weights: "
            f"date={pd.Timestamp(bad.get('date')).strftime('%Y-%m-%d')}, industry_code={bad.get('industry_code')}"
        )

    panel["weight"] = panel["paid_hours"] / panel["total_paid_hours"]
    result = panel.sort_values(["date", "industry_code"]).reset_index(drop=True)
    return result


def reconcile_headline_earnings(component_panel, weights, benchmark):
    """Reconstruct headline AHE from component weights and report the residual."""
    if component_panel is None or weights is None or benchmark is None:
        raise ValueError("component_panel, weights, and benchmark cannot be None")
    if not isinstance(component_panel, pd.DataFrame):
        raise TypeError("component_panel must be a pandas DataFrame")
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("weights must be a pandas DataFrame")
    if not isinstance(benchmark, pd.DataFrame):
        raise TypeError("benchmark must be a pandas DataFrame")

    required_component = {"date", "industry_code", "average_hourly_earnings"}
    missing_component = sorted(required_component - set(component_panel.columns))
    if missing_component:
        raise ValueError(
            f"component_panel is missing required columns for earnings reconciliation: {missing_component}"
        )
    if {"date", "industry_code", "weight"} - set(weights.columns):
        raise ValueError("weights must include columns: date, industry_code, weight")
    if {
        "date",
        "published_total_private_ahe",
        "published_total_private_employment",
    } - set(benchmark.columns):
        raise ValueError(
            "benchmark must include columns: date, published_total_private_ahe, published_total_private_employment"
        )

    component = component_panel.copy()
    component["date"] = pd.to_datetime(component["date"]).dt.normalize()
    weight_df = weights.copy()
    weight_df["date"] = pd.to_datetime(weight_df["date"]).dt.normalize()
    benchmark_df = benchmark.copy()
    benchmark_df["date"] = pd.to_datetime(benchmark_df["date"]).dt.normalize()
    required_benchmark = [
        "date",
        "published_total_private_ahe",
        "published_total_private_employment",
    ]
    benchmark_df = benchmark_df[
        [col for col in required_benchmark if col in benchmark_df.columns]
    ].copy()

    if set(required_benchmark) - set(benchmark_df.columns):
        missing = sorted(set(required_benchmark) - set(benchmark_df.columns))
        raise ValueError(f"benchmark must include columns: {missing}")

    if benchmark_df["date"].duplicated().any():
        dupes = benchmark_df.loc[benchmark_df["date"].duplicated(), "date"].unique()
        raise ValueError(
            f"duplicate benchmark rows for dates: {sorted(pd.to_datetime(dupes).strftime('%Y-%m-%d').tolist())}"
        )

    joined = component[
        ["date", "industry_code", "average_hourly_earnings", "employment"]
    ].merge(
        weight_df[["date", "industry_code", "weight"]],
        on=["date", "industry_code"],
        how="left",
    )
    missing_weights = joined[joined["weight"].isna()]
    if not missing_weights.empty:
        bad = missing_weights.iloc[0]
        raise ValueError(
            "missing weight for component row: "
            f"date={pd.Timestamp(bad.get('date')).strftime('%Y-%m-%d')}, industry_code={bad.get('industry_code')}"
        )

    joined["weighted_hourly_earnings"] = (
        joined["weight"] * joined["average_hourly_earnings"]
    )
    reconstructed = (
        joined.groupby("date", as_index=False)["weighted_hourly_earnings"]
        .sum()
        .rename(columns={"weighted_hourly_earnings": "reconstructed_ahe"})
    )

    merged = reconstructed.merge(benchmark_df, on="date", how="left")
    if merged["published_total_private_ahe"].isna().any():
        missing = merged.loc[merged["published_total_private_ahe"].isna(), "date"].iloc[
            0
        ]
        raise ValueError(
            f"benchmark date missing for reconciliation: {pd.Timestamp(missing).strftime('%Y-%m-%d')}"
        )

    merged["earnings_residual"] = (
        merged["published_total_private_ahe"] - merged["reconstructed_ahe"]
    )

    component_employment = (
        component.groupby("date", as_index=False)["employment"]
        .sum()
        .rename(columns={"employment": "component_employment"})
    )
    merged = merged.merge(component_employment, on="date", how="left")
    merged["employment_residual"] = (
        merged["published_total_private_employment"] - merged["component_employment"]
    )

    output = (
        merged[
            [
                "date",
                "published_total_private_ahe",
                "reconstructed_ahe",
                "earnings_residual",
                "published_total_private_employment",
                "component_employment",
                "employment_residual",
            ]
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )
    return output


def _serialize_observations(observations):
    records = []
    for row in observations.to_dict(orient="records"):
        obs_date = row["date"]
        if isinstance(obs_date, pd.Timestamp):
            date_str = obs_date.strftime("%Y-%m-%d")
        else:
            date_str = pd.Timestamp(obs_date).strftime("%Y-%m-%d")

        records.append(
            {
                "series_id": row["series_id"],
                "date": date_str,
                "value": float(row["value"]),
                "footnotes": row.get("footnotes") or [],
            }
        )

    return records


def save_snapshot(output_dir, observations, metadata, requests_payloads, series_spec):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc)
    snapshot_path = output_path / _snapshot_filename(now_utc)
    if snapshot_path.exists():
        raise FileExistsError(f"Snapshot path already exists: {snapshot_path}")

    document = {
        "format_version": _SNAPSHOT_FORMAT_VERSION,
        "source": metadata["source"],
        "endpoint": metadata["endpoint"],
        "retrieval_start_utc": metadata["retrieval_start_utc"],
        "retrieval_end_utc": metadata["retrieval_end_utc"],
        "series_spec": series_spec,
        "requests": requests_payloads,
        "observations": _serialize_observations(observations),
    }

    temp_path = output_path / f".{snapshot_path.name}.tmp"
    try:
        with temp_path.open("x", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True, separators=(",", ":"))
        temp_path.replace(snapshot_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    return snapshot_path


def _validate_snapshot_document(document, snapshot_path):
    if not isinstance(document, dict):
        raise ValueError(f"Snapshot is not a JSON object: {snapshot_path}")

    required_fields = {
        "format_version",
        "source",
        "endpoint",
        "retrieval_start_utc",
        "retrieval_end_utc",
        "series_spec",
        "requests",
        "observations",
    }
    missing = sorted(required_fields - set(document.keys()))
    if missing:
        raise ValueError(f"Snapshot missing required fields {missing}: {snapshot_path}")

    if document["format_version"] != _SNAPSHOT_FORMAT_VERSION:
        raise ValueError(
            "Unsupported snapshot format version "
            f"{document['format_version']} in {snapshot_path}"
        )


def load_snapshot(snapshot_path):
    input_path = Path(snapshot_path)
    try:
        with input_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Snapshot JSON is malformed: {input_path}") from exc

    _validate_snapshot_document(document, input_path)

    observations = pd.DataFrame(
        document["observations"], columns=["series_id", "date", "value", "footnotes"]
    )
    if not observations.empty:
        observations["date"] = pd.to_datetime(observations["date"])
        observations["value"] = observations["value"].astype(float)
        observations["footnotes"] = observations["footnotes"].apply(
            lambda x: x if isinstance(x, list) else []
        )

    metadata = {
        "format_version": document["format_version"],
        "source": document["source"],
        "endpoint": document["endpoint"],
        "retrieval_start_utc": document["retrieval_start_utc"],
        "retrieval_end_utc": document["retrieval_end_utc"],
        "requests": document["requests"],
        "series_spec": document["series_spec"],
    }

    return observations, metadata


def refresh_data(snapshot_dir, start_year=2019, end_year=None, *, as_of_year=None):
    effective_end_year = (
        end_year
        if end_year is not None
        else as_of_year
        if as_of_year is not None
        else date.today().year
    )
    if effective_end_year < start_year:
        raise ValueError(
            f"Invalid year range: start_year {start_year} is after end_year {effective_end_year}."
        )

    payloads = build_request_payloads(start_year, effective_end_year)
    if not payloads:
        raise ValueError("No request payloads generated for the selected year range.")

    refresh_start_utc = datetime.now(timezone.utc)
    provenance = []
    frames = []
    all_series_ids = [record["series_id"] for record in SERIES_SPEC]

    try:
        for payload in payloads:
            envelope = fetch_bls_batch(payload)
            batch_series_ids = payload["seriesid"]
            table = parse_bls_response(
                envelope,
                batch_series_ids,
                start_year,
                effective_end_year,
            )
            frames.append(table)
            provenance.append({"payload": payload, "response": envelope})
    except Exception:
        raise

    if not frames:
        raise BLSRequestError(
            "No response data was collected for the requested refresh."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["series_id", "date"]).reset_index(drop=True)
    observed_ids = sorted(combined["series_id"].dropna().unique().tolist())
    expected_ids = sorted(all_series_ids)
    missing_ids = [
        series_id for series_id in expected_ids if series_id not in observed_ids
    ]
    unexpected_ids = [
        series_id for series_id in observed_ids if series_id not in expected_ids
    ]
    if missing_ids or unexpected_ids:
        raise BLSRequestError(
            "Refresh result does not match the requested series set: "
            f"missing={missing_ids}, unexpected={unexpected_ids}"
        )

    metadata = {
        "format_version": _SNAPSHOT_FORMAT_VERSION,
        "retrieval_start_utc": refresh_start_utc.isoformat().replace("+00:00", "Z"),
        "retrieval_end_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "source": "bls",
        "endpoint": _BLS_V1_ENDPOINT,
    }

    snapshot_path = save_snapshot(
        snapshot_dir,
        combined,
        metadata,
        provenance,
        SERIES_SPEC,
    )
    return snapshot_path
