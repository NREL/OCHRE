# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Pull building files, weather, and EPlus results from a ResStock output into the golden test directory.

Set resstock_output_directory below to point to your ResStock simulation output,
then run:
    uv run test/resstock_golden/copy_eplus_result.py
"""

import csv
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

resstock_output_directory = "/Users/radhikar/Documents/buildstock2025/res_ochre/resstock/national_baseline_super_ochre"

# --- Paths (relative to this script) ---
SCRIPT_DIR = Path(__file__).resolve().parent
BUILDINGS_DIR = SCRIPT_DIR / "eplus_result"
WEATHER_DIR = SCRIPT_DIR / "weather"
EPLUS_CSV = SCRIPT_DIR / "eplus_result" / "eplus_annual_result.csv"

# JSON section name -> CSV column prefix.  Sections not listed here are skipped.
SECTION_PREFIX = {
    "BuildExistingModel": "build_existing_model",
    "QOIReport": "qoi_report",
    "ReportSimulationOutput": "report_simulation_output",
    "ReportUtilityBills": "report_utility_bills",
    "UpgradeCosts": "upgrade_costs",
}

# ResStock weather directory (sibling of the output directory's parent project)
RESSTOCK_WEATHER_DIR = Path(resstock_output_directory).resolve().parent.parent / "weather"


def discover_runs(output_dir):
    """Scan runN/ directories and return {building_id: run_path} for successful runs."""
    runs = {}
    skipped = []
    output_path = Path(output_dir)

    for entry in sorted(output_path.iterdir()):
        if not entry.is_dir() or not re.match(r"run\d+$", entry.name):
            continue

        run_path = entry / "run"
        json_path = run_path / "data_point_out.json"

        if not json_path.exists():
            skipped.append(entry.name)
            continue

        with open(json_path) as f:
            data = json.load(f)

        bem = data.get("BuildExistingModel")
        if not bem or "building_id" not in bem:
            skipped.append(entry.name)
            continue

        bldg_id = int(bem["building_id"])
        runs[bldg_id] = entry

    return runs, skipped


def copy_building_files(bldg_id, run_dir):
    """Copy home.xml and in.schedules.csv for a building."""
    bldg_name = f"bldg{bldg_id:07d}"
    dest_dir = BUILDINGS_DIR / bldg_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    run_path = run_dir / "run"
    for filename in ("home.xml", "in.schedules.csv"):
        src = run_path / filename
        if src.exists():
            shutil.copy2(src, dest_dir / filename)
        else:
            print(f"  WARNING: {src} not found for {bldg_name}")


def extract_weather_filename(home_xml_path):
    """Parse <EPWFilePath> from home.xml and return just the filename."""
    try:
        tree = ET.parse(home_xml_path)
        # EPWFilePath can be under various namespaces; search broadly
        for elem in tree.iter():
            if elem.tag.endswith("EPWFilePath") and elem.text:
                return os.path.basename(elem.text.strip())
    except ET.ParseError:
        pass
    return None


def copy_weather_files(runs):
    """Copy unique weather files referenced by buildings."""
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    weather_files = set()

    for bldg_id in sorted(runs):
        bldg_name = f"bldg{bldg_id:07d}"
        home_xml = BUILDINGS_DIR / bldg_name / "home.xml"
        if not home_xml.exists():
            continue

        epw_name = extract_weather_filename(home_xml)
        if epw_name and epw_name not in weather_files:
            weather_files.add(epw_name)
            src = RESSTOCK_WEATHER_DIR / epw_name
            if src.exists():
                shutil.copy2(src, WEATHER_DIR / epw_name)
            else:
                print(f"  WARNING: Weather file {src} not found")

    return weather_files


def build_eplus_csv(runs):
    """Build eplus_annual_result.csv from data_point_out.json and out.osw files."""
    rows = []
    all_columns = set()

    for bldg_id in sorted(runs):
        run_dir = runs[bldg_id]
        run_path = run_dir / "run"

        with open(run_path / "data_point_out.json") as f:
            data = json.load(f)

        # Read metadata from out.osw
        osw_path = run_dir / "out.osw"
        osw = {}
        if osw_path.exists():
            with open(osw_path) as f:
                osw = json.load(f)

        row = {
            "building_id": bldg_id,
            "job_id": int(re.search(r"(\d+)$", run_dir.name).group(1)),
            "started_at": osw.get("started_at", ""),
            "completed_at": osw.get("completed_at", ""),
            "completed_status": osw.get("completed_status", ""),
        }

        # Add empty apply_upgrade columns (baseline run)
        for suffix in ("applicable", "upgrade_name", "reference_scenario"):
            row[f"apply_upgrade.{suffix}"] = ""

        # Flatten JSON sections into prefixed columns
        for section, prefix in SECTION_PREFIX.items():
            section_data = data.get(section, {})
            for key, value in section_data.items():
                col = f"{prefix}.{key}"
                row[col] = value

        all_columns.update(row.keys())
        rows.append(row)

    # Determine column order: fixed columns first, then sorted section columns
    fixed_columns = [
        "building_id",
        "job_id",
        "started_at",
        "completed_at",
        "completed_status",
    ]
    section_columns = sorted(all_columns - set(fixed_columns))
    fieldnames = fixed_columns + section_columns

    with open(EPLUS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return len(rows), len(fieldnames)


def cleanup_old_buildings(current_ids):
    """Remove building directories and orphaned weather files not in the current set."""
    current_names = {f"bldg{bid:07d}" for bid in current_ids}
    removed = []

    if BUILDINGS_DIR.exists():
        for entry in sorted(BUILDINGS_DIR.iterdir()):
            if entry.is_dir() and entry.name.startswith("bldg") and entry.name not in current_names:
                shutil.rmtree(entry)
                removed.append(entry.name)

    # Collect weather files still referenced by remaining buildings
    used_weather = set()
    for name in current_names:
        home_xml = BUILDINGS_DIR / name / "home.xml"
        if home_xml.exists():
            epw = extract_weather_filename(home_xml)
            if epw:
                used_weather.add(epw)

    removed_weather = []
    if WEATHER_DIR.exists():
        for entry in sorted(WEATHER_DIR.iterdir()):
            if entry.suffix == ".epw" and entry.name not in used_weather:
                entry.unlink()
                removed_weather.append(entry.name)

    return removed, removed_weather


def main():
    output_dir = Path(resstock_output_directory)
    if not output_dir.is_dir():
        print(f"ERROR: ResStock output directory not found: {output_dir}")
        return

    print(f"ResStock output: {output_dir}")
    print(f"Golden test dir: {SCRIPT_DIR}")

    # 1. Discover buildings
    runs, skipped = discover_runs(output_dir)
    print(f"\nDiscovered {len(runs)} successful buildings, {len(skipped)} skipped")
    for s in skipped:
        print(f"  Skipped: {s}")

    # 2. Copy building files
    print("\nCopying building files...")
    for bldg_id in sorted(runs):
        copy_building_files(bldg_id, runs[bldg_id])
    print(f"  Copied home.xml + in.schedules.csv for {len(runs)} buildings")

    # 3. Copy weather files
    print("\nCopying weather files...")
    weather_files = copy_weather_files(runs)
    print(f"  Copied {len(weather_files)} unique weather files")

    # 4. Build EPlus CSV
    print("\nBuilding eplus_annual_result.csv...")
    n_rows, n_cols = build_eplus_csv(runs)
    print(f"  Written {n_rows} rows x {n_cols} columns to {EPLUS_CSV}")

    # 5. Cleanup
    removed_bldgs, removed_weather = cleanup_old_buildings(set(runs.keys()))
    if removed_bldgs:
        print(f"\nRemoved {len(removed_bldgs)} old building directories:")
        for name in removed_bldgs:
            print(f"  {name}")
    if removed_weather:
        print(f"Removed {len(removed_weather)} orphaned weather files:")
        for name in removed_weather:
            print(f"  {name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
