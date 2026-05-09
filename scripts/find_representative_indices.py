"""Script to find representative indices for each model's design preferences."""

import polars as pl

from design_bias.analysis import extract_features
from prismal.config import DATA_DIR, OUTPUT_DIR


def main():
    experiment = "pgais-pilot"
    results_dir = OUTPUT_DIR / experiment
    result_files = list(results_dir.glob(f"{experiment}_*.parquet"))

    # Load prompts
    df_prompts = pl.read_parquet(DATA_DIR / "prompts-lite-dataset.parquet")

    all_data = []
    for f in result_files:
        model_id = f.stem.replace(f"{experiment}_", "")
        df = pl.read_parquet(f)

        # We need to keep track of the list index before exploding
        df = df.with_columns(
            [
                pl.lit(model_id).alias("model"),
                pl.col("responses")
                .map_elements(
                    lambda x: list(range(len(x))), return_dtype=pl.List(pl.Int64)
                )
                .alias("response_idx"),
            ]
        )

        # Explode responses and response_idx together
        df = df.explode(["responses", "response_idx"])
        all_data.append(df)

    df_analysis = pl.concat(all_data).rename({"responses": "code"})
    df_analysis = df_analysis.join(df_prompts, on="index")

    # Extract features
    df_analysis = extract_features(df_analysis)

    # Filter for Latent + Professional (good baseline for style)
    df_latent = df_analysis.filter(
        (pl.col("framework").is_null()) & (pl.col("descriptor") == "professional")
    )

    models = df_analysis["model"].unique().to_list()

    print(
        f"{'Model':<20} | {'Index':<6} | {'RespIdx':<8} | {'FW':<12} | {'Font':<15} | {'Palette'}"
    )
    print("-" * 85)

    for model in models:
        m_df = df_latent.filter(pl.col("model") == model)

        # For CodeLlama, look for the 167-declaration template specifically
        if model == "codellama_latest":
            # Let's look for a Retro one since we know it uses the pixel font there
            m_df = df_analysis.filter(
                (pl.col("model") == model)
                & (pl.col("descriptor") == "retro")
                & (pl.col("framework").is_null())
            )
            pref_df = m_df.filter(pl.col("css_decl_count") == 167)
            if not pref_df.is_empty():
                m_df = pref_df
        elif model == "gpt-oss_120b":
            # GPT often uses Bootstrap + Blue as a default
            # Let's try to find a blue one
            pref_df = m_df.filter(pl.col("detected_palette").str.contains("blue"))
            if not pref_df.is_empty():
                m_df = pref_df

        # Sort by index to get a stable representative
        m_df = m_df.sort("index")

        # Pick the most "representative" (first in the filtered set)
        if not m_df.is_empty():
            row = m_df.head(1).to_dicts()[0]
            print(
                f"{model:<20} | {row['index']:<6} | {row['response_idx']:<8} | {row['detected_framework']:<12} | {row['detected_font']:<15} | {row['detected_palette']}"
            )


if __name__ == "__main__":
    main()
