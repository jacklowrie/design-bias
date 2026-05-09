import marimo

__generated_with = "0.23.4"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl

    return Path, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # Parquet Viewer
    """)
    return


@app.cell
def _(mo):
    path_input = mo.ui.text(
        label="Parquet File Path (relative to project root)",
        placeholder="outputs/pgais-pilot/pgais-pilot_gpt-oss_120b.parquet",
        value="outputs/pgais-pilot/pgais-pilot_gpt-oss_120b.parquet",
    )
    path_input
    return (path_input,)


@app.cell
def _(Path, mo, path_input, pl):
    path = Path(path_input.value)

    if not path.exists():
        mo.md(f"**Error:** File not found at `{path}`")

    try:
        df = pl.read_parquet(path)

        # Analysis for response counts
        if "responses" in df.columns:
            df_with_count = df.with_columns(
                response_count=pl.col("responses").list.len()
            )
            invalid_rows = df_with_count.filter(pl.col("response_count") < 50)

            analysis_result = mo.vstack(
                [
                    mo.md("### Analysis: `responses` count"),
                    mo.md(f"Total rows: {len(df)}"),
                    mo.md(f"Rows with < 50 responses: {len(invalid_rows)}"),
                ]
            )

            if len(invalid_rows) > 0:
                analysis_result = mo.vstack(
                    [
                        analysis_result,
                        mo.md("#### Invalid Rows (count < 50):"),
                        mo.ui.table(invalid_rows),
                    ]
                )
            else:
                analysis_result = mo.vstack(
                    [analysis_result, mo.md("✅ All rows have at least 50 responses.")]
                )
        else:
            analysis_result = mo.md(
                "⚠️ Column `responses` not found in dataframe for analysis."
            )

        result = mo.vstack(
            [
                mo.md(f"### Data Preview: `{path}`"),
                mo.md(f"**Shape:** {df.shape}"),
                mo.ui.table(df),
                analysis_result,
            ]
        )
    except Exception as e:
        result = mo.md(f"**Error reading parquet:** {e!s}")
    df.head()
    return (df,)


@app.cell
def _(Path, df, pl):
    df.height
    prompts = pl.read_parquet(Path.cwd() / "data" / "prompts-lite-dataset.parquet")
    prompts.height
    for row in df.iter_rows(named=True):
        print(len(row["responses"]))
    return


if __name__ == "__main__":
    app.run()
