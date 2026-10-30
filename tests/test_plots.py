"""Tests of the figures, on records written by hand."""

from typing import Any

import numpy as np
from matplotlib.collections import LineCollection

from decoder_fixes_gross.plots import floor_against_fig1


def _entry(
    p: float, occurrence: tuple[float | None, float | None, float | None], exact: float
) -> dict[str, Any]:
    value, low, high = occurrence
    return {
        "p": p,
        "occurrence": {"per_round": {"value": value, "low": low, "high": high}},
        "exact_event": {"per_round": {"value": exact, "low": exact, "high": exact}},
    }


# The table rate 3e-4 is not on the curve grid; at 4e-3 the occurrence floor reached 1.
RECORD = {
    "p_grids": {"table": [3e-4], "curve": [1e-4, 1e-3, 4e-3]},
    "floor": [
        _entry(1e-4, (1e-12, 0.8e-12, 1.2e-12), 1e-13),
        _entry(3e-4, (5e-11, 4e-11, 6e-11), 2e-12),
        _entry(1e-3, (1e-8, 0.8e-8, 1.2e-8), 2e-13),
        _entry(4e-3, (None, 1e-3, None), 1e-15),
    ],
}


def test_lines_are_the_per_round_floors_on_the_curve_grid() -> None:
    occurrence, exact = floor_against_fig1(RECORD).axes[0].get_lines()
    np.testing.assert_array_equal(occurrence.get_xdata(), [1e-4, 1e-3])
    np.testing.assert_array_equal(occurrence.get_ydata(), [1e-12, 1e-8])
    np.testing.assert_array_equal(exact.get_xdata(), [1e-4, 1e-3, 4e-3])
    np.testing.assert_array_equal(exact.get_ydata(), [1e-13, 2e-13, 1e-15])


def test_axes_are_log_log() -> None:
    ax = floor_against_fig1(RECORD).axes[0]
    assert (ax.get_xscale(), ax.get_yscale()) == ("log", "log")


def test_band_spans_the_low_and_high_ends_where_both_exist() -> None:
    ax = floor_against_fig1(RECORD).axes[0]
    band = next(c for c in ax.collections if c.get_label() == "Occurrence floor -+ error")
    vertices = np.asarray(band.get_paths()[0].vertices, dtype=np.float64)
    # The polygon runs along the low ends and back along the high ends.
    for p, low, high in [(1e-4, 0.8e-12, 1.2e-12), (1e-3, 0.8e-8, 1.2e-8)]:
        at_p = vertices[np.isclose(vertices[:, 0], p, rtol=1e-12, atol=0)][:, 1]
        assert np.isclose(at_p.min(), low, rtol=1e-12, atol=0)
        assert np.isclose(at_p.max(), high, rtol=1e-12, atol=0)
    assert not np.any(np.isclose(vertices[:, 0], 4e-3, rtol=1e-12, atol=0))


def test_readings_are_drawn_at_the_two_fig1_rates() -> None:
    ax = floor_against_fig1(RECORD).axes[0]
    readings = next(c for c in ax.collections if c.get_label() == "Fig. 1 estimate, as read")
    assert isinstance(readings, LineCollection)
    segments = sorted(segment.tolist() for segment in readings.get_segments())
    assert segments == [[[4e-4, 3e-10], [4e-4, 4e-10]], [[1e-3, 1e-8], [1e-3, 2e-8]]]
