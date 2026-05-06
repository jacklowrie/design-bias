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
        placeholder="outputs/prompts.parquet",
        value="outputs/prompts.parquet",
    )
    path_input
    return (path_input,)


@app.cell
def _(Path, mo, path_input, pl):
    path = Path(path_input.value)

    if not path.exists():
        result = mo.md(f"**Error:** File not found at `{path}`")
    else:
        try:
            df = pl.read_parquet(path)
            result = mo.vstack(
                [
                    mo.md(f"### Data Preview: `{path}`"),
                    mo.md(f"**Shape:** {df.shape}"),
                    mo.ui.table(df),
                ]
            )
        except Exception as e:
            result = mo.md(f"**Error reading parquet:** {e!s}")

    result
    return


if __name__ == "__main__":
    app.run()
