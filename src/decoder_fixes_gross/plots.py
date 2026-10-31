"""Figures drawn from the aggregate records in ``results/``.

Figures are built with matplotlib's object API (:class:`matplotlib.figure.Figure`), never through
pyplot, so no global figure state is created and no backend is selected.
"""

from collections.abc import Mapping
from typing import Any

from matplotlib.figure import Figure

# Categorical slots 1 and 2 of the reference palette, and neutral ink for the readings.
_OCCURRENCE_COLOR = "#2a78d6"
_EXACT_EVENT_COLOR = "#eb6834"
_READING_COLOR = "#3d3d3a"
# Bottom of the floor axis: the exact-event floor falls far below every other quantity at
# large p, and a full range would flatten the readings and the band.
_Y_BOTTOM = 1e-16


def floor_against_fig1(floor_record: Mapping[str, Any], gate_record: Mapping[str, Any]) -> Figure:
    """Return the per-round floor over the curve grid, with the Fig. 1 readings beside it.

    Parameters
    ----------
    floor_record : Mapping
        A floor record as written to ``results/floor_<run_id>.json`` by the aggregation
        command: its ``p_grids.curve`` and ``floor`` entries are read.
    gate_record : Mapping
        A gate record as written to ``results/gate_relay_bp.json``: the ``p`` and ``reading``
        of each ``floor_check`` entry are read.

    Returns
    -------
    matplotlib.figure.Figure
        One log-log axes. Its first line is the occurrence floor per round, its second the
        exact-event floor; the band around the occurrence floor spans its per-round low and
        high ends; the vertical segments are the ranges of the gate record's floor check, read
        off Fig. 1 of arXiv:2608.11516.
        A point whose per-round value is null is left out of its line, and a null end leaves
        that rate out of the band. The vertical axis is cut at 1e-16.
    """
    curve = set(floor_record["p_grids"]["curve"])
    entries = [entry for entry in floor_record["floor"] if entry["p"] in curve]

    def points(form: str, key: str) -> tuple[list[float], list[float]]:
        found = [(e["p"], e[form]["per_round"][key]) for e in entries]
        kept = [(p, value) for p, value in found if value is not None]
        return [p for p, _ in kept], [value for _, value in kept]

    figure = Figure(figsize=(6.4, 4.4), layout="constrained")
    ax = figure.add_subplot()
    ax.plot(
        *points("occurrence", "value"),
        color=_OCCURRENCE_COLOR,
        linewidth=2,
        marker="o",
        markersize=4,
        label="Occurrence floor (headline)",
    )
    ax.plot(
        *points("exact_event", "value"),
        color=_EXACT_EVENT_COLOR,
        linewidth=2,
        linestyle="--",
        label="Exact-event floor, Eq. (1)",
    )
    band = [
        (e["p"], e["occurrence"]["per_round"]["low"], e["occurrence"]["per_round"]["high"])
        for e in entries
        if e["occurrence"]["per_round"]["low"] is not None
        and e["occurrence"]["per_round"]["high"] is not None
    ]
    ax.fill_between(
        [p for p, _, _ in band],
        [low for _, low, _ in band],
        [high for _, _, high in band],
        color=_OCCURRENCE_COLOR,
        alpha=0.2,
        linewidth=0,
        label="Occurrence floor -+ error",
    )
    readings = [
        (point["p"], point["reading"]["low"], point["reading"]["high"])
        for point in gate_record["floor_check"]
    ]
    ax.vlines(
        [p for p, _, _ in readings],
        [low for _, low, _ in readings],
        [high for _, _, high in readings],
        color=_READING_COLOR,
        linewidth=2,
        zorder=3,
        label="Fig. 1 estimate, as read",
    )
    # Each reading is short on the log axis and sits on the occurrence line, so a hollow marker
    # at its geometric middle shows where it is.
    ax.scatter(
        [p for p, _, _ in readings],
        [(low * high) ** 0.5 for _, low, high in readings],
        s=90,
        marker="D",
        facecolors="none",
        edgecolors=_READING_COLOR,
        linewidths=1.5,
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=_Y_BOTTOM)
    ax.set_xlabel("Physical error rate p")
    ax.set_ylabel("Logical error rate per round")
    ax.grid(visible=True, which="major", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    return figure
