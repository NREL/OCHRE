"""
Output control utilities for managing verbosity-based output filtering.

This module provides functions to expand template patterns and build the
enabled_outputs set used throughout OCHRE for determining which outputs
to generate.
"""

from ochre.defaults.output_registry import (
    OUTPUT_REGISTRY,
    END_USES,
    ZONE_NAMES,
    BOUNDARY_NAMES,
)


def expand_template(pattern, placeholders):
    """
    Expand a template pattern with all placeholder combinations.

    Parameters
    ----------
    pattern : str
        Pattern with {placeholder} syntax, e.g., "{end_use} Electric Power (kW)"
    placeholders : dict
        Mapping of placeholder names to lists of values

    Returns
    -------
    list[str]
        All expanded output names

    Examples
    --------
    >>> expand_template("{end_use} SOC (-)", {"end_use": ["EV", "Battery"]})
    ["EV SOC (-)", "Battery SOC (-)"]
    """
    if "{" not in pattern:
        return [pattern]

    results = [pattern]

    for placeholder, values in placeholders.items():
        key = "{" + placeholder + "}"
        new_results = []
        for r in results:
            if key in r:
                for val in values:
                    new_results.append(r.replace(key, val))
            else:
                new_results.append(r)
        results = new_results

    return results


def get_enabled_outputs(output_format, verbosity):
    """
    Get the set of all enabled output names for a given format and verbosity.

    Expands all template patterns using the superset of all possible values
    defined in the output registry. The resulting set is used for O(1) lookup
    when generating results.

    Parameters
    ----------
    output_format : str
        Output format ('ochre' or 'resstock')
    verbosity : int
        Verbosity level (0-9)

    Returns
    -------
    frozenset[str]
        Immutable set of enabled output column names

    Examples
    --------
    >>> enabled = get_enabled_outputs('ochre', 3)
    >>> "Temperature - Indoor (C)" in enabled
    True
    >>> "EV SOC (-)" in enabled
    True
    """
    registry = OUTPUT_REGISTRY.get(output_format, OUTPUT_REGISTRY["ochre"])

    placeholders = {
        "end_use": END_USES,
        "results_name": END_USES,
        "zone_name": ZONE_NAMES,
        "boundary_name": BOUNDARY_NAMES,
    }

    enabled = set()

    # Add outputs for all levels up to and including verbosity
    for level in range(verbosity + 1):
        patterns = registry.get(level, [])
        for pattern in patterns:
            expanded = expand_template(pattern, placeholders)
            enabled.update(expanded)

    return frozenset(enabled)
