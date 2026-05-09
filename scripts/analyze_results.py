"""Consolidated script to analyze experiment results for design_bias."""

import argparse

import polars as pl
from loguru import logger

from design_bias.analysis import (
    extract_features,
    run_complexity_stats,
    run_significance_test,
)
from prismal.config import DATA_DIR, OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze design_bias experiment results."
    )
    parser.add_argument(
        "-e",
        "--experiment",
        default="pgais-pilot",
        help="Name of the experiment to analyze (default: pgais-pilot).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Fallback to pgais-debug if pilot results are missing.",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_args()
    experiment = args.experiment
    results_dir = OUTPUT_DIR / experiment

    # Discovery logic
    result_files = list(results_dir.glob(f"{experiment}_*.parquet"))

    if not result_files and (args.debug or experiment == "pgais-pilot"):
        logger.warning(
            f"No results found for {experiment}, falling back to pgais-debug"
        )
        experiment = "pgais-debug"
        results_dir = OUTPUT_DIR / experiment
        result_files = list(results_dir.glob(f"{experiment}_*.parquet"))

    if not result_files:
        logger.error(f"No result files found in {results_dir}")
        return

    # Load prompts
    prompts_path = DATA_DIR / "prompts-lite-dataset.parquet"
    df_prompts = pl.read_parquet(prompts_path)

    # Load results
    logger.info(f"Loading results for experiment: {experiment}")
    dfs = []
    for f in result_files:
        model_id = f.stem.replace(f"{experiment}_", "")
        df_res = pl.read_parquet(f)
        df_res = df_res.with_columns(pl.lit(model_id).alias("model"))
        dfs.append(df_res)

    df_all = pl.concat(dfs)
    df_master = df_all.join(df_prompts, on="index")
    df_analysis = df_master.explode("responses").rename({"responses": "code"})

    logger.info(f"Extracting features from {len(df_analysis)} generations...")
    df_analysis = extract_features(df_analysis)

    # 1. Framework Analysis
    logger.info("\n" + "=" * 50)
    logger.info(f" ANALYSIS SUMMARY: {experiment.upper()} ")
    logger.info("=" * 50)

    # 2. Complexity Analysis
    run_complexity_stats(df_analysis)

    # 3. Latent Significance Analysis
    df_latent = df_analysis.filter(pl.col("framework").is_null())
    if not df_latent.is_empty():
        logger.info("\n--- LATENT DESIGN PREFERENCES (Control Group) ---")

        # Baseline: Strictly latent (no framework, no descriptor)
        df_strict_latent = df_latent.filter(pl.col("descriptor").is_null())
        if not df_strict_latent.is_empty():
            logger.info("\n--- STRICT LATENT (No Framework, No Descriptor) ---")
            run_significance_test(
                df_strict_latent,
                "detected_framework",
                "Framework Choice",
                expected_categories=["tailwind", "bootstrap", "bulma", "custom/none"],
            )
            run_significance_test(df_strict_latent, "detected_font", "Font Choice")
            run_significance_test(
                df_strict_latent, "detected_palette", "Palette Choice"
            )

        # Descriptor Breakdown
        df_with_descriptors = df_latent.filter(pl.col("descriptor").is_not_null())
        if not df_with_descriptors.is_empty():
            logger.info(
                "\n--- DESCRIPTOR BREAKDOWN (No Framework, With Descriptor) ---"
            )
            run_significance_test(
                df_with_descriptors,
                "detected_framework",
                "Framework Choice",
                group_by="descriptor",
                expected_categories=["tailwind", "bootstrap", "bulma", "custom/none"],
            )
            run_significance_test(
                df_with_descriptors,
                "detected_font",
                "Font Choice",
                group_by="descriptor",
            )
            run_significance_test(
                df_with_descriptors,
                "detected_palette",
                "Palette Choice",
                group_by="descriptor",
            )

        # Model Breakdown for Latent Preferences
        logger.info("\n--- MODEL BREAKDOWN (Latent Preferences) ---")
        run_significance_test(
            df_latent,
            "detected_framework",
            "Framework Choice",
            group_by="model",
            expected_categories=["tailwind", "bootstrap", "bulma", "custom/none"],
        )
        run_significance_test(
            df_latent,
            "detected_font",
            "Font Choice",
            group_by="model",
        )
        run_significance_test(
            df_latent,
            "detected_palette",
            "Palette Choice",
            group_by="model",
        )
    else:
        logger.info("\nNo latent (control) samples found.")

    # 4. Adherence Analysis
    df_req = df_analysis.filter(pl.col("framework").is_not_null())
    if not df_req.is_empty():
        logger.info("\n--- MODEL ADHERENCE ---")
        df_adherence = df_req.with_columns(
            (pl.col("framework") == pl.col("detected_framework")).alias("followed")
        )
        logger.info(
            f"\n{
                df_adherence.group_by('model').agg(
                    pl.col('followed').mean().alias('adherence_rate')
                )
            }"
        )

    # 5. Complexity by Model
    logger.info("\n--- COMPLEXITY BY MODEL ---")
    run_complexity_stats(df_analysis, group_by="model")


if __name__ == "__main__":
    main()
