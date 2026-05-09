"""Script to generate sample images of model designs using html2pic."""

import polars as pl
from html2pic import Html2Pic
from loguru import logger

from prismal.config import DATA_DIR, OUTPUT_DIR


def main() -> None:
    """Main execution function."""
    experiment = "pgais-pilot"
    results_dir = OUTPUT_DIR / experiment
    samples_dir = OUTPUT_DIR / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts to get metadata
    prompts_path = DATA_DIR / "prompts-lite-dataset.parquet"
    if not prompts_path.exists():
        logger.error(f"Prompts not found at {prompts_path}")
        return
    df_prompts = pl.read_parquet(prompts_path)

    # Discovery logic
    result_files = list(results_dir.glob(f"{experiment}_*.parquet"))
    if not result_files:
        logger.warning(f"No results found for {experiment}, checking debug")
        experiment = "pgais-debug"
        results_dir = OUTPUT_DIR / experiment
        result_files = list(results_dir.glob(f"{experiment}_*.parquet"))

    if not result_files:
        logger.error(f"No result files found in {results_dir}")
        return

    # Load and join data
    dfs = []
    for f in result_files:
        model_id = f.stem.replace(f"{experiment}_", "")
        df_res = pl.read_parquet(f)
        df_res = df_res.with_columns(pl.lit(model_id).alias("model"))
        dfs.append(df_res)

    df_all = pl.concat(dfs)
    df_master = df_all.join(df_prompts, on="index")
    df_analysis = df_master.explode("responses").rename({"responses": "code"})

    # Filter for representative designs: Latent + Professional
    # We want a design where the model chose everything itself but had a prompt theme
    df_reps = df_analysis.filter(
        (pl.col("framework").is_null()) & (pl.col("descriptor") == "professional")
    )

    # If we don't have enough, just take the first generation per model
    models = df_analysis["model"].unique().to_list()

    logger.info(f"Generating samples for models: {models}")

    for model in models:
        # Try to find a professional latent design first
        model_df = df_reps.filter(pl.col("model") == model)
        if model_df.is_empty():
            # Fallback to any design
            model_df = df_analysis.filter(pl.col("model") == model)

        if model_df.is_empty():
            continue

        # Take the first one
        sample = model_df.head(1).to_dicts()[0]
        html_code = sample["code"]

        output_path = samples_dir / f"{model}_sample.png"
        logger.info(f"Rendering sample for {model} to {output_path}...")

        try:
            # html2pic rendering
            # We set a reasonable size for a settings page
            hp = Html2Pic(html_code, width=1024)
            hp.save(str(output_path))
            logger.info(f"Successfully saved {output_path}")
        except Exception as e:
            logger.error(f"Failed to render {model}: {e}")


if __name__ == "__main__":
    main()
