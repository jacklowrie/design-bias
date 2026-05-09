import polars as pl

from design_bias.analysis import extract_features
from prismal.config import DATA_DIR, OUTPUT_DIR

EXPERIMENT_NAME = "pgais-pilot"
PROMPTS_PATH = DATA_DIR / "prompts-lite-dataset.parquet"

df_prompts = pl.read_parquet(PROMPTS_PATH)
results_dir = OUTPUT_DIR / EXPERIMENT_NAME
for model_name in ["codellama_latest", "gpt-oss_120b", "llama4_latest"]:
    f = results_dir / f"{EXPERIMENT_NAME}_{model_name}.parquet"
    if f.exists():
        df_res = pl.read_parquet(f)
        df_master = df_res.join(df_prompts, on="index")
        df_analysis = df_master.explode("responses").rename({"responses": "code"})
        df_analysis = extract_features(df_analysis)

        df_specified = df_analysis.filter(pl.col("framework").is_not_null())
        total_specified = len(df_specified)

        adherent = df_specified.filter(
            pl.col("framework") == pl.col("detected_framework")
        )
        count_adherent = len(adherent)
        rate = count_adherent / total_specified if total_specified > 0 else 0

        print(f"Model: {model_name}")
        print(f"  Total specified: {total_specified}")
        print(f"  Adherent: {count_adherent}")
        print(f"  Rate: {rate:.6f}")

        if model_name == "codellama_latest":
            non_adherent = df_specified.filter(
                pl.col("framework") != pl.col("detected_framework")
            )
            breakdown = (
                non_adherent.group_by(["framework", "detected_framework"])
                .len()
                .sort("len", descending=True)
            )
            print("\n  CodeLlama Non-Adherence Breakdown:")
            print(breakdown)

            # Sample one specific divergence
            example = non_adherent.filter(
                (pl.col("framework") == "tailwind")
                & (pl.col("detected_framework") == "bootstrap")
            ).head(1)
            if not example.is_empty():
                print("\n  Example: Requested Tailwind, got Bootstrap (Head of code):")
                print(example["code"].item()[:500])
    else:
        print(f"Result file for {model_name} not found.")
