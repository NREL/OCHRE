"""Conftest for test_dwelling: collects EPlus cross-validation discrepancies
and prints per-metric summary tables at the end of the pytest session."""

from __future__ import annotations

import math

import pytest

_EPLUS_KEY = "_eplus_discrepancies"


def pytest_configure(config):
    """Initialize the discrepancy collector on the pytest Config object."""
    setattr(config, _EPLUS_KEY, {})


@pytest.fixture
def eplus_collector(request):
    """Provide tests access to the shared EPlus discrepancy dict."""
    return getattr(request.config, _EPLUS_KEY)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print one table per metric that had EPlus discrepancies."""
    from test.test_dwelling.test_resstock_golden import SUMMARY_COLUMNS

    discrepancies = getattr(config, _EPLUS_KEY, {})
    if not discrepancies:
        return

    char_headers = [hdr for _, hdr in SUMMARY_COLUMNS]

    terminalreporter.write_line("")
    terminalreporter.section("EPlus Cross-Validation Summary", sep="=")

    for metric in sorted(discrepancies):
        rows = discrepancies[metric]
        if not rows:
            continue

        # Sort by abs(%diff) descending
        rows.sort(key=lambda r: abs(r[3]) if math.isfinite(r[3]) else float("inf"), reverse=True)

        # Compute dynamic column widths for characteristics
        char_widths = {}
        for hdr in char_headers:
            vals = [r[4].get(hdr, "") for r in rows]
            char_widths[hdr] = max(len(hdr), max((len(v) for v in vals), default=0))

        # Build header
        char_hdr_parts = "  ".join(f"{hdr:<{char_widths[hdr]}}" for hdr in char_headers)
        char_sep_parts = "  ".join("-" * char_widths[hdr] for hdr in char_headers)

        terminalreporter.write_line("")
        terminalreporter.write_line(f"  {metric}")
        terminalreporter.write_line(
            f"  {'Building':<16}{char_hdr_parts}  {'OCHRE':>12}  {'EPlus':>12}  {'%Diff':>10}"
        )
        terminalreporter.write_line(
            f"  {'-' * 16}{char_sep_parts}  {'-' * 12}  {'-' * 12}  {'-' * 10}"
        )

        for bldg_name, ochre_val, eplus_val, pct_diff, chars in rows:
            char_vals = "  ".join(
                f"{chars.get(hdr, ''):<{char_widths[hdr]}}" for hdr in char_headers
            )
            if math.isfinite(pct_diff):
                pct_str = f"{pct_diff:>+9.1f}%"
            else:
                pct_str = f"{'inf':>10}"
            terminalreporter.write_line(
                f"  {bldg_name:<16}{char_vals}  {ochre_val:>12.3f}  {eplus_val:>12.3f}  {pct_str}"
            )

    terminalreporter.write_line("")
