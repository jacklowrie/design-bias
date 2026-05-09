from pathlib import Path

import polars as pl


def extract_and_save():
    output_dir = Path("outputs")
    pilot_dir = output_dir / "pgais-pilot"

    # Representative samples mapping
    # (filename_stem, row_index, response_index, output_name)
    samples = [
        ("pgais-pilot_llama4_latest", 4, 0, "representative_llama4.txt"),
        ("pgais-pilot_gpt-oss_120b", 4, 2, "representative_gpt-oss.txt"),
        ("pgais-pilot_codellama_latest", 12, 0, "representative_codellama.txt"),
    ]

    for stem, row_idx, resp_idx, out_name in samples:
        file_path = pilot_dir / f"{stem}.parquet"
        if not file_path.exists():
            print(f"Error: {file_path} not found")
            continue

        df = pl.read_parquet(file_path)
        # Filter by index column (assuming the 'index' column matches the indices found earlier)
        # Our previous tool output showed these are the absolute indices in the dataframe.
        # But to be safe, we'll check the 'index' column if it exists, otherwise use row number.

        row = (
            df.filter(pl.col("index") == row_idx)
            if "index" in df.columns
            else df.slice(row_idx, 1)
        )

        if row.is_empty():
            print(f"Error: Could not find row {row_idx} in {stem}")
            continue

        responses = row["responses"].item()
        if resp_idx >= len(responses):
            print(
                f"Error: Response index {resp_idx} out of range for {stem} at row {row_idx}"
            )
            continue

        html_content = responses[resp_idx]

        output_path = output_dir / out_name
        with open(output_path, "w") as f:
            f.write(html_content)
        print(f"Saved {out_name} to {output_path}")


if __name__ == "__main__":
    extract_and_save()
