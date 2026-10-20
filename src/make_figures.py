"""Create the report's SVG figures from saved results.

Run: python src/make_figures.py
Writes figures under results/figures/.
"""

import argparse
import math
from pathlib import Path

import pandas as pd


RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURES = RESULTS / "figures"
SUFFIX = ""


def bounds(low: float, high: float, step: float) -> tuple[float, float, list[float]]:
    """Axis range covering the data, rounded out to whole steps."""
    lo = math.floor(low / step) * step
    hi = math.ceil(high / step) * step
    n = int(round((hi - lo) / step))
    return lo, hi, [round(lo + i * step, 2) for i in range(n + 1)]


MONTHS = {6: "June", 7: "July", 8: "August", 9: "September"}
MODELS = {
    "random_forest": ("Random forest", "circle", ""),
    "hist_gradient_boosting": ("Histogram gradient boosting", "square", "6 4"),
}
ARMS = {
    "same_year_oracle": ("Final same-year crop mask", "#13795b"),
    "previous_year": ("Previous-year crop mask", "#bc5b21"),
}
# Font sizes in pixels; figures shrink to about half size at page width.
LABEL_SIZE = 19
HEADING_SIZE = 21
SPACE = 10  # Space between a label and the thing it labels.
RATIO_LABEL = "RMSE ratio (crop mask / cropland mask)"
# Room on the left for the tick numbers and the y-axis label.
AXIS_ROOM = 70 + LABEL_SIZE + SPACE


def text(x: float, y: float, value: str, size: int = LABEL_SIZE, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="#222">{value}</text>'
    )


def y_axis_label(middle: float) -> str:
    x = SPACE + LABEL_SIZE
    return (
        f'<text x="{x}" y="{middle:.1f}" transform="rotate(-90 {x} {middle:.1f})" '
        f'font-family="Arial, sans-serif" font-size="{LABEL_SIZE}" '
        f'text-anchor="middle" fill="#222">{RATIO_LABEL}</text>'
    )


def point(x: float, y: float, shape: str, color: str) -> str:
    if shape == "square":
        return f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" fill="{color}"/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'


def border(left: float, top: float, width: float, height: float) -> str:
    """Light frame so each panel title clearly belongs to the box under it."""
    return f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="none" stroke="#bbb"/>'


def sample(x: float, y: float, color: str, shape: str, dash: str) -> list[str]:
    """Short piece of a plotted line, for the legend."""
    dash_value = f' stroke-dasharray="{dash}"' if dash else ""
    return [
        f'<line x1="{x}" y1="{y}" x2="{x + 35}" y2="{y}" stroke="{color}" stroke-width="2"{dash_value}/>',
        point(x + 18, y, shape, color),
    ]


def legend_grid(left: float, top: float, step: float, arms: dict) -> list[str]:
    """Legend as a small grid: masks across the top, models down the side, and
    each cell is a piece of the real line."""
    name_room = 15 * LABEL_SIZE
    column_step = 14 * LABEL_SIZE  # room for the longer mask names
    parts = []
    for column, (arm_name, _) in enumerate(arms.values()):
        x = left + name_room + column * column_step
        parts.append(text(x + 17, top + LABEL_SIZE / 3, arm_name, anchor="middle"))
    for row, (model_name, shape, dash) in enumerate(MODELS.values(), start=1):
        y = top + row * step
        parts.append(text(left, y + LABEL_SIZE / 3, model_name))
        for column, (_, color) in enumerate(arms.values()):
            parts.extend(sample(left + name_room + column * column_step, y, color, shape, dash))
    return parts


def timing_figure() -> None:
    data = pd.read_csv(RESULTS / f"masks_by_month{SUFFIX}.csv")
    data = data[data["arm"].isin(ARMS)]
    panel_width, panel_height = 455, 245
    lefts = [AXIS_ROOM, AXIS_ROOM + panel_width + 7 * SPACE]
    # Leave room between the upper month labels and the lower panel titles.
    first_top = HEADING_SIZE + 2 * SPACE
    tops = [first_top, first_top + panel_height + LABEL_SIZE + HEADING_SIZE + 6 * SPACE]
    # Separate the legend from the month labels.
    legend_top = tops[1] + panel_height + LABEL_SIZE + 5 * SPACE
    legend_step = round(LABEL_SIZE * 1.6)
    width = lefts[1] + panel_width + 2 * SPACE
    height = legend_top + 2 * legend_step + LABEL_SIZE + SPACE
    y_min, y_max, ticks = bounds(min(data["q_vs_cropland"].min(), 1),
                                 max(data["q_vs_cropland"].max(), 1), 0.04)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        y_axis_label((tops[0] + tops[1] + panel_height) / 2),
    ]

    for row, crop in enumerate(["CORN", "SOYBEANS"]):
        for column, evaluation in enumerate(["temporal_holdout", "county_heldout_cv"]):
            left, top = lefts[column], tops[row]
            bottom = top + panel_height
            right = left + panel_width
            # Months sit inside the panel so "September" does not run past its edge.
            x_for = lambda month: left + (month - 5.5) * panel_width / 4
            y_for = lambda value: bottom - (value - y_min) * panel_height / (y_max - y_min)
            title = "future years" if evaluation == "temporal_holdout" else "unseen counties"
            parts.append(text((left + right) / 2, top - SPACE, f"{crop.title()}: {title}",
                              HEADING_SIZE, "middle"))
            for value in ticks:
                y = y_for(value)
                parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#dddddd"/>')
                parts.append(text(left - SPACE, y + LABEL_SIZE / 3, f"{value:.2f}", LABEL_SIZE, "end"))
            for month, name in MONTHS.items():
                parts.append(text(x_for(month), bottom + LABEL_SIZE + SPACE, name, LABEL_SIZE, "middle"))
            parts.append(f'<line x1="{left}" y1="{y_for(1):.1f}" x2="{right}" y2="{y_for(1):.1f}" stroke="#444"/>')
            parts.append(border(left, top, panel_width, panel_height))

            panel = data[(data["crop"] == crop) & (data["evaluation"] == evaluation)]
            for arm, (_, color) in ARMS.items():
                for model, (_, shape, dash) in MODELS.items():
                    values = panel[(panel["arm"] == arm) & (panel["model"] == model)].sort_values("cutoff_month")
                    coordinates = [(x_for(item.cutoff_month), y_for(item.q_vs_cropland)) for item in values.itertuples()]
                    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in coordinates)
                    dash_value = f' stroke-dasharray="{dash}"' if dash else ""
                    parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"{dash_value}/>')
                    parts.extend(point(x, y, shape, color) for x, y in coordinates)

    parts.extend(legend_grid(lefts[0], legend_top, legend_step, ARMS))
    parts.append('</svg>')
    (FIGURES / f"masks_by_month{SUFFIX}.svg").write_text("\n".join(parts), encoding="utf-8")


def state_figure() -> None:
    data = pd.read_csv(RESULTS / f"masks_by_state{SUFFIX}.csv")
    data = data[
        (data["crop"] == "CORN") & (data["cutoff_month"] == 9)
        & (data["arm"] == "same_year_oracle")
    ].copy()
    order = data.groupby("state")["q_vs_cropland"].mean().sort_values().index.tolist()
    left, top, chart_width, chart_height = AXIS_ROOM, 2 * SPACE, 800, 370
    width = left + chart_width + 2 * SPACE
    bottom = top + chart_height
    middle = left + chart_width / 2
    legend_y = bottom + LABEL_SIZE + 5 * SPACE
    height = legend_y + LABEL_SIZE + SPACE
    y_min, y_max, ticks = bounds(min(data["q_vs_cropland_lo"].min(), 1),
                                 max(data["q_vs_cropland_hi"].max(), 1), 0.1)
    x_for = lambda index: left + (index + 0.5) * chart_width / len(order)
    y_for = lambda value: bottom - (value - y_min) * chart_height / (y_max - y_min)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        y_axis_label((top + bottom) / 2),
    ]
    for value in ticks:
        y = y_for(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(text(left - SPACE, y + LABEL_SIZE / 3, f"{value:.1f}", LABEL_SIZE, "end"))
    parts.append(f'<line x1="{left}" y1="{y_for(1):.1f}" x2="{left + chart_width}" y2="{y_for(1):.1f}" stroke="#444"/>')
    parts.append(border(left, top, chart_width, chart_height))

    styles = {"random_forest": ("#286090", -6), "hist_gradient_boosting": ("#b24c34", 6)}
    for model, (color, offset) in styles.items():
        panel = data[data["model"] == model].set_index("state").loc[order]
        for index, state in enumerate(order):
            row = panel.loc[state]
            x = x_for(index) + offset
            low, high, value = map(y_for, [row.q_vs_cropland_lo, row.q_vs_cropland_hi, row.q_vs_cropland])
            parts.append(f'<line x1="{x:.1f}" y1="{low:.1f}" x2="{x:.1f}" y2="{high:.1f}" stroke="{color}"/>')
            parts.append(f'<line x1="{x - 4:.1f}" y1="{low:.1f}" x2="{x + 4:.1f}" y2="{low:.1f}" stroke="{color}"/>')
            parts.append(f'<line x1="{x - 4:.1f}" y1="{high:.1f}" x2="{x + 4:.1f}" y2="{high:.1f}" stroke="{color}"/>')
            parts.append(point(x, value, "circle" if model == "random_forest" else "square", color))
    for index, state in enumerate(order):
        parts.append(text(x_for(index), bottom + LABEL_SIZE + SPACE, state, LABEL_SIZE, "middle"))
    # The two legend entries start a quarter and a half of the way across.
    for x, (model, (color, _)) in zip([left + chart_width / 4, middle], styles.items()):
        shape = "circle" if model == "random_forest" else "square"
        parts.append(point(x, legend_y, shape, color))
        parts.append(text(x + 12, legend_y + LABEL_SIZE / 3, MODELS[model][0]))
    parts.append('</svg>')
    (FIGURES / f"september_corn_by_state{SUFFIX}.svg").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    global SUFFIX
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", choices=["MOD13A3", "MOD13Q1"], default="MOD13Q1")
    SUFFIX = {"MOD13A3": "", "MOD13Q1": "_250m"}[parser.parse_args().product]
    FIGURES.mkdir(exist_ok=True)
    timing_figure()
    state_figure()
    print(f"Wrote SVG figures under {FIGURES}")


if __name__ == "__main__":
    main()
