"""Analysis utilities for design_bias experiments."""

import re
from typing import Any

import polars as pl
from loguru import logger
from scipy.stats import chisquare, kruskal, mannwhitneyu

# Color classification utilities
COLOR_MAP = {
    "blue": (59, 130, 246),
    "gray": (107, 114, 128),
    "slate": (71, 85, 105),
    "red": (239, 68, 68),
    "green": (34, 197, 94),
    "emerald": (16, 185, 129),
    "indigo": (99, 102, 241),
    "violet": (139, 92, 246),
    "amber": (245, 158, 11),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}


def detect_framework(code: str) -> str:
    """Detect the CSS framework used in the code."""
    code_lower = code.lower()
    if "cdn.tailwindcss.com" in code_lower or "tailwindcss" in code_lower:
        return "tailwind"
    if "bootstrap" in code_lower:
        return "bootstrap"
    if "bulma" in code_lower:
        return "bulma"
    return "custom/none"


def extract_primary_font(code: str) -> str:
    """Extract the primary font family from the CSS."""
    fonts = re.findall(r"font-family:\s*([^;!}\n]+)", code)
    if fonts:
        # Cleanup: remove quotes, take first in list
        return fonts[0].split(",")[0].replace("'", "").replace('"', "").strip()
    return "unknown"


def detect_visual_styles(code: str) -> str:
    """Detect visual features like borders, shadows, and rounded corners."""
    styles = []
    code_lower = code.lower()

    if (
        "border:" in code_lower
        or "border-width" in code_lower
        or "border-style" in code_lower
    ):
        styles.append("borders")
    if "border-bottom" in code_lower or "<hr" in code_lower:
        styles.append("dividers")
    if "box-shadow" in code_lower or "shadow-" in code_lower:
        styles.append("shadows")
    if "border-radius" in code_lower or "rounded" in code_lower:
        styles.append("rounded-corners")

    return ", ".join(styles) if styles else "minimal"


def analyze_field_order(code: str) -> str:
    """Analyze the appearance order of key settings fields."""
    fields = ["Email", "Password", "Username", "Notifications", "Theme"]
    positions = []
    for f in fields:
        pos = code.lower().find(f.lower())
        if pos != -1:
            positions.append((f, pos))

    ordered = [p[0] for p in sorted(positions, key=lambda x: x[1])]
    return " -> ".join(ordered) if ordered else "unknown"


def hex_to_rgb(hex_str: str) -> tuple[int, int, int] | None:
    """Convert hex color string to RGB tuple."""
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join([c * 2 for c in hex_str])
    try:
        return (
            int(hex_str[0:2], 16),
            int(hex_str[2:4], 16),
            int(hex_str[4:6], 16),
        )
    except (ValueError, IndexError):
        return None


def color_distance(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    """Calculate Euclidean distance between two RGB colors."""
    return sum((a - b) ** 2 for a, b in zip(c1, c2, strict=True)) ** 0.5


def classify_color(hex_str: str) -> str:
    """Classify a hex color into a standard design color name."""
    rgb = hex_to_rgb(hex_str)
    if not rgb:
        return "unknown"
    best_match = "unknown"
    min_dist = float("inf")
    for name, ref_rgb in COLOR_MAP.items():
        dist = color_distance(rgb, ref_rgb)
        if dist < min_dist:
            min_dist = dist
            best_match = name
    return best_match


def detect_palette(code: str) -> str:
    """Extract and classify the dominant color palette."""
    hex_colors = list(set(re.findall(r"#[0-9a-fA-F]{3,6}", code)))
    classifications = [classify_color(h) for h in hex_colors]
    brand_colors = [
        c for c in classifications if c not in ["white", "black", "unknown"]
    ]
    if not brand_colors:
        return "monochrome"
    counts: dict[str, int] = {}
    for c in brand_colors:
        counts[c] = counts.get(c, 0) + 1
    sorted_colors = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)
    return "/".join(sorted_colors[:2])


def detect_css_strategy(code: str) -> tuple[str, int]:
    """Detect the CSS strategy (framework vs hybrid) and count declarations."""
    code_lower = code.lower()
    fw = None
    if "cdn.tailwindcss.com" in code_lower or "tailwindcss" in code_lower:
        fw = "tailwind"
    elif "bootstrap" in code_lower:
        fw = "bootstrap"
    elif "bulma" in code_lower:
        fw = "bulma"

    style_blocks = re.findall(r"<style[^>]*>(.*?)</style>", code, re.DOTALL)
    has_custom = False
    css_decl_count = 0
    if style_blocks:
        content = "".join(style_blocks).strip()
        css_decl_count = content.count(";")
        if css_decl_count > 0:
            has_custom = True

    strategy = "minimal/none"
    if fw and has_custom:
        strategy = f"hybrid ({fw})"
    elif fw:
        strategy = f"framework ({fw})"
    elif has_custom:
        strategy = "custom only"

    return strategy, css_decl_count


def run_significance_test(
    df: pl.DataFrame,
    column: str,
    title: str,
    group_by: str | None = None,
    expected_categories: list[str] | None = None,
) -> dict[str, Any]:
    """Run a Chi-Square significance test on a column, optionally grouped."""
    if group_by:
        logger.info(f"\n--- Significance Test for {title} (grouped by {group_by}) ---")
        results = {}
        for val in df[group_by].unique():
            group_df = df.filter(pl.col(group_by) == val)
            if group_df.is_empty():
                continue
            logger.info(f"\nGroup: {val}")
            results[val] = run_significance_test(
                group_df,
                column,
                f"{title} [{val}]",
                expected_categories=expected_categories,
            )
        return results

    logger.info(f"\n--- Significance Test for {title} ---")

    # If expected_categories are provided, ensure all are present (even with 0 counts)
    if expected_categories:
        counts_df = df.group_by(column).len()
        # Create a dataframe with all expected categories
        full_categories = pl.DataFrame({column: expected_categories})
        counts = (
            full_categories.join(counts_df, on=column, how="left")
            .fill_null(0)
            .sort("len", descending=True)
        )
    else:
        counts = df.group_by(column).len().sort("len", descending=True)

    logger.info("Observed Frequencies:")
    logger.info(f"\n{counts}")

    observed = counts["len"].to_list()
    if len(observed) < 2:
        logger.info("Not enough variation to perform Chi-Square test.")
        return {"chi2": None, "p": None, "counts": counts, "significant": False}

    chi2, p = chisquare(observed)
    logger.info(f"Chi-Square Statistic: {chi2:.2f}")
    logger.info(f"P-value: {p:.4g}")

    significant = p < 0.05
    if significant:
        logger.info("Result: STATISTICALLY SIGNIFICANT (p < 0.05). Reject H0.")
        logger.info(f"The model exhibits a non-random preference in {title}.")
    else:
        logger.info("Result: NOT SIGNIFICANT. Fail to reject H0.")

    return {"chi2": chi2, "p": p, "counts": counts, "significant": significant}


def run_complexity_stats(
    df: pl.DataFrame, group_by: str = "detected_framework"
) -> dict[str, Any]:
    """Run complexity analysis (Kruskal-Wallis) on CSS declarations."""
    logger.info(f"\n--- Customization Complexity (grouped by {group_by}) ---")
    summary = (
        df.group_by(group_by)
        .agg(
            [
                pl.col("css_decl_count").mean().alias("mean_decls"),
                pl.col("css_decl_count").median().alias("median_decls"),
                pl.len().alias("n"),
            ]
        )
        .sort("mean_decls", descending=True)
    )
    logger.info(f"\n{summary}")

    groups_vals = [val for val in df[group_by].unique() if val is not None]
    if group_by == "detected_framework":
        groups_vals = [v for v in groups_vals if v != "custom/none"]

    groups = [
        df.filter(pl.col(group_by) == val)["css_decl_count"].to_list()
        for val in groups_vals
    ]

    results: dict[str, Any] = {"summary": summary, "h_stat": None, "p_val": None}

    if len(groups) > 1:
        try:
            h_stat, p_val = kruskal(*groups)
            results["h_stat"] = h_stat
            results["p_val"] = p_val
            logger.info(
                "\nKruskal-Wallis H-test (Global Difference): H={:.2f}, p={:.4g}",
                h_stat,
                p_val,
            )
        except ValueError as e:
            logger.warning(f"Could not run Kruskal-Wallis: {e}")
            return results

        if p_val < 0.05:
            logger.info(
                "RESULT: Significant difference between groups. Pairwise tests:"
            )
            # Only do specific pairwise for framework if that's what we are grouping by
            if (
                group_by == "detected_framework"
                and "tailwind" in groups_vals
                and "bootstrap" in groups_vals
            ):
                t_data = df.filter(pl.col("detected_framework") == "tailwind")[
                    "css_decl_count"
                ]
                b_data = df.filter(pl.col("detected_framework") == "bootstrap")[
                    "css_decl_count"
                ]
                _, p_pair = mannwhitneyu(t_data, b_data)
                results["p_pairwise"] = {"tailwind_vs_bootstrap": p_pair}
                logger.info(f"Tailwind vs Bootstrap: p={p_pair:.4g}")
        else:
            logger.info("RESULT: No statistically significant difference in volume.")

    return results


def extract_features(df: pl.DataFrame) -> pl.DataFrame:
    """Apply all feature extraction functions to a dataframe with a 'code' column."""

    def row_extractor(row: tuple[Any, ...]) -> tuple[Any, ...]:
        code = row[0]
        strategy, decl_count = detect_css_strategy(code)
        return (
            detect_framework(code),
            extract_primary_font(code),
            detect_visual_styles(code),
            analyze_field_order(code),
            detect_palette(code),
            strategy,
            decl_count,
        )

    extractions = df.select(pl.col("code")).map_rows(row_extractor)
    extractions.columns = [
        "detected_framework",
        "detected_font",
        "detected_visuals",
        "detected_order",
        "detected_palette",
        "detected_strategy",
        "css_decl_count",
    ]

    # Drop existing columns if they exist to avoid duplicates before concat
    cols_to_drop = [c for c in extractions.columns if c in df.columns]
    return pl.concat([df.drop(cols_to_drop), extractions], how="horizontal")
