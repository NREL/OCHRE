"""ResStock output format utilities for OCHRE."""

import datetime as dt
import os
import re

import pandas as pd

from ochre.utils.base import default_input_path
from ochre.utils.units import convert


def to_underscore_case(s):
    """Port of OpenStudio's toUnderscoreCase (utilities/core/String.cpp).

    Converts arbitrary strings (including camelCase, digit boundaries, and
    special characters) into lower_snake_case.  Used by ResStock to derive
    CSV column names from metric display names.
    """
    # Collapse brand names so camelCase splitting doesn't insert underscores
    # (matches the C++ replace_all calls in toUnderscoreCase).
    result = s.replace("OpenStudio", "Openstudio").replace("EnergyPlus", "Energyplus")
    result = re.sub(r"[^a-zA-Z0-9]", " ", result)
    result = re.sub(r"[-]+", "_", result)
    result = re.sub(r"\s+", "_", result)
    result = re.sub(r"([A-Za-z])([0-9])", r"\1_\2", result)
    result = re.sub(r"([0-9]+)([A-Za-z])", r"\1_\2", result)
    result = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", result)
    result = re.sub(r"([a-z])([A-Z])", r"\1_\2", result)
    return result.lower().strip("_")


def parse_unit(col_name):
    """Extract unit from parenthesized suffix, e.g. 'Power (kW)' -> 'kW'."""
    start = col_name.rfind("(")
    end = col_name.rfind(")")
    if start != -1 and end != -1 and end > start:
        return col_name[start + 1 : end]
    return ""


def convert_units(value, from_unit, to_unit, hours_per_step=1.0):
    """Convert value between OCHRE and ResStock units.

    Uses pint for unit conversion factors. Power-to-energy conversions
    (kW->kWh, etc.) multiply by hours_per_step since pint can't handle
    the implicit time integration.
    """
    if from_unit == to_unit or not from_unit or not to_unit:
        return value
    key = (from_unit, to_unit)
    if key == ("kW", "kWh"):
        return value * hours_per_step
    if key == ("W", "kWh"):
        return value * 0.001 * hours_per_step
    if key == ("W", "kBtu"):
        return value * 0.001 * hours_per_step * convert(1, "kW", "kBtu/hr")
    if key == ("therms/hour", "kBtu"):
        return value * hours_per_step * convert(1, "therm", "kBtu")
    if key == ("therms/hour", "kWh"):
        return value * hours_per_step * convert(1, "therm", "kWh")
    if key == ("C", "F"):
        # Temperature is an offset conversion; pint can't handle pandas Series
        # for offset units, so apply the formula directly.
        return value * (convert(1, "delta_degC", "delta_degF")) + convert(0, "degC", "degF")
    if key == ("m^3/s", "cfm"):
        return value * convert(1, "m^3/s", "cubic_feet/min")
    if key == ("kWh", "MBtu"):
        return value * convert(1, "kWh", "MBtu")
    if key == ("kBtu", "MBtu"):
        return value * convert(1, "kBtu", "MBtu")
    return value


def load_crosswalk(crosswalk_file=None):
    """Load the ResStock-OCHRE crosswalk CSV."""
    if crosswalk_file is None:
        crosswalk_file = os.path.join(default_input_path, "resstock_ochre_crosswalk.csv")
    return pd.read_csv(crosswalk_file)


def build_resstock_timeseries(df, crosswalk, time_res):
    """Convert OCHRE DataFrame to ResStock timeseries format."""
    hours_per_step = time_res.total_seconds() / 3600

    # Filter crosswalk to rows with both OCHRE and Timeseries columns
    valid_mappings = crosswalk[
        crosswalk["OCHRE"].notna()
        & (crosswalk["OCHRE"] != "")
        & crosswalk["ResStock Timeseries"].notna()
        & (crosswalk["ResStock Timeseries"] != "")
    ].copy()

    result = pd.DataFrame(index=df.index)
    units_dict = {}

    for _, row in valid_mappings.iterrows():
        ochre_col = row["OCHRE"]
        resstock_col = row["ResStock Timeseries"]
        target_unit = row.get("ResStock Timeseries Unit", "")
        if pd.isna(target_unit):
            target_unit = ""

        if ochre_col not in df.columns:
            continue

        from_unit = parse_unit(ochre_col)
        result[resstock_col] = convert_units(df[ochre_col], from_unit, target_unit, hours_per_step)
        units_dict[resstock_col] = target_unit

    return result, units_dict


def _build_ts_to_annual(crosswalk):
    """Build mapping from ResStock Timeseries column to ResStock Annual column."""
    ts_to_annual = {}
    for _, row in crosswalk.iterrows():
        ts = row.get("ResStock Timeseries", "")
        annual = row.get("ResStock Annual", "")
        if pd.notna(ts) and ts and pd.notna(annual) and annual:
            ts_to_annual[ts] = annual
    return ts_to_annual


def accumulate_annual_sums(resstock_df, units_dict, crosswalk, existing_sums=None):
    """Accumulate ResStock timeseries column sums, converted to annual units."""
    sums = existing_sums.copy() if existing_sums else {}
    ts_to_annual = _build_ts_to_annual(crosswalk)

    for col in resstock_df.columns:
        if col not in ts_to_annual:
            continue
        annual_col = ts_to_annual[col]
        ts_unit = units_dict.get(col, "")
        annual_unit = parse_unit(annual_col)
        chunk_sum = convert_units(resstock_df[col].sum(), ts_unit, annual_unit)
        sums[annual_col] = sums.get(annual_col, 0) + chunk_sum

    return sums


def write_resstock_timeseries(df, units_dict, file_path, append=False):
    """Write ResStock timeseries CSV with header + units rows."""
    if df is None or df.empty:
        return

    df_out = df.reset_index().rename(columns={"index": "Time"})

    if append and os.path.exists(file_path):
        df_out.to_csv(file_path, index=False, header=False, mode="a")
    else:
        units_row = [""] + [units_dict.get(col, "") for col in df.columns]
        with open(file_path, "w") as f:
            f.write(",".join(["Time"] + list(df.columns)) + "\n")
            f.write(",".join(units_row) + "\n")
        df_out.to_csv(file_path, index=False, header=False, mode="a")


def write_resstock_annual(sums, file_path):
    """Write accumulated annual totals to results_annual.csv."""
    if not sums:
        return
    with open(file_path, "w") as f:
        for key, value in sums.items():
            f.write(f"{key},{round(value, 3)}\n")
    return pd.DataFrame(list(sums.items()), columns=["Metric", "Value"])


def _get_resstock_agg_func(col, units_dict):
    """Determine aggregation function for hourly resampling based on unit."""
    unit = units_dict.get(col, "")
    # Energy/quantity columns should be summed; temperature/rate columns averaged
    sum_units = {"kWh", "kBtu", "lb", "gal", "hr"}
    return "sum" if unit in sum_units else "mean"


class ResStockOutput:
    """Manages ResStock output file generation (timeseries + annual)."""

    def __init__(self, output_path, time_res):
        self.timeseries_file = os.path.join(output_path, "results_timeseries.csv")
        self.annual_file = os.path.join(output_path, "results_annual.csv")
        self.crosswalk = load_crosswalk()
        self.time_res = time_res
        self._units_dict = None
        self._annual_sums = {}

        if os.path.exists(self.timeseries_file):
            os.remove(self.timeseries_file)

    def export_chunk(self, df):
        """Convert OCHRE chunk to ResStock format, accumulate annual sums, write timeseries."""
        resstock_df, units_dict = build_resstock_timeseries(df, self.crosswalk, self.time_res)

        if self._units_dict is None:
            self._units_dict = units_dict

        self._annual_sums = accumulate_annual_sums(resstock_df, self._units_dict, self.crosswalk, self._annual_sums)

        append = os.path.exists(self.timeseries_file)
        write_resstock_timeseries(resstock_df, self._units_dict, self.timeseries_file, append=append)

    def finalize(self, df=None, failed=False):
        """Write final chunk, annual totals, and return hourly resampled data."""
        if df is not None and not failed:
            self.export_chunk(df)
        annual_df = write_resstock_annual(self._annual_sums, self.annual_file)

        if failed or not os.path.exists(self.timeseries_file):
            return pd.DataFrame(), annual_df, pd.DataFrame()

        # Read back full timeseries, skipping units row (row index 1)
        ts_df = pd.read_csv(self.timeseries_file, skiprows=[1], parse_dates=["Time"], index_col="Time")

        # Resample to hourly
        agg_funcs = {col: _get_resstock_agg_func(col, self._units_dict) for col in ts_df.columns}
        hourly_df = ts_df.resample(dt.timedelta(hours=1)).agg(agg_funcs)

        return ts_df, annual_df, hourly_df
