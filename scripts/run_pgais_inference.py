"""Script to run inference on PGAIS using AsyncPGAISInferenceClient."""

import argparse
import asyncio
from pathlib import Path

import polars as pl
from loguru import logger
from tqdm.asyncio import tqdm

from design_bias.config import DesignPromptDataset
from design_bias.pgai_inference import AsyncPGAISInferenceClient
from prismal.cli import add_common_arguments
from prismal.config import ROOT_DIR, ConfigBase
from prismal.data import ResponsesDatasetBase, ResponsesRowBase


async def run_inference(
    config: ConfigBase,
    input_path: Path,
    output_dir: Path,
    model_id: str | None = None,
    temperature: float | None = None,
    max_concurrency: int | None = None,
    rpm_limit: int | None = None,
    max_retries: int | None = None,
) -> None:
    """Run inference using AsyncPGAISInferenceClient.

    Args:
        config: The experiment configuration.
        input_path: Path to the input parquet file.
        output_dir: Path to the directory where results will be saved.
        model_id: Model ID to use (overrides config).
        temperature: Temperature for sampling (overrides config).
        max_concurrency: Max concurrency (overrides config).
        rpm_limit: Max requests per minute (overrides config).
        max_retries: Max retries (overrides config).
    """
    # Determine models to use
    models_to_use = [model_id] if model_id else config.get_model_ids()
    temp_to_use = temperature if temperature is not None else config.model.temperature
    num_samples = config.model.num_samples
    concurrency_to_use = (
        max_concurrency
        if max_concurrency is not None
        else config.compute.max_concurrency
    )
    rpm_to_use = (
        rpm_limit if rpm_limit is not None else config.compute.requests_per_minute
    )
    retries_to_use = (
        max_retries if max_retries is not None else config.compute.max_retries
    )

    # Load and validate input data
    input_df = pl.read_parquet(input_path)
    # Convert to list of dicts for Pydantic validation
    input_data = input_df.to_dicts()
    try:
        from design_bias.config import DesignPromptRow

        prompts = [DesignPromptRow(**d) for d in input_data]
        prompt_dataset = DesignPromptDataset(prompts=prompts)
        logger.info(
            "Loaded and validated {} prompts from {}",
            len(prompt_dataset.prompts),
            input_path,
        )
    except Exception as e:
        logger.error(
            "Failed to validate input prompts against DesignPromptDataset: {}", e
        )
        raise

    # Prepare client
    client = AsyncPGAISInferenceClient(
        max_retries=retries_to_use,
        requests_per_minute=rpm_to_use,
    )

    for model_to_use in models_to_use:
        logger.info(
            "Running inference with model: {} ({} samples, temp: {})",
            model_to_use,
            num_samples,
            temp_to_use,
        )

        # Prepare batch prompts
        batch_prompts = []
        batch_system_prompts = []
        prompt_indices = []

        for prompt_row in prompt_dataset.prompts:
            for _ in range(num_samples):
                batch_prompts.append(prompt_row.prompt)
                batch_system_prompts.append(prompt_row.system_prompt)
                prompt_indices.append(prompt_row.index)

        # Run batch inference
        logger.debug(
            "Starting batch inference for model {} with {} prompts",
            model_to_use,
            len(batch_prompts),
        )
        responses = await client.generate_batch(
            model_id=model_to_use,
            prompts=batch_prompts,
            system_prompts=batch_system_prompts,
            temperature=temp_to_use,
            seed=config.model.seed,
            max_concurrency=concurrency_to_use,
            rpm_limit=rpm_to_use,
            show_progress=True,
        )

        # Extract text from InferenceResponse objects
        response_texts = []
        for r in responses:
            if isinstance(r, BaseException):
                logger.error("Inference failed for a prompt: {}", r)
                response_texts.append("")
            else:
                response_texts.append(r.content)

        # Organize responses into ResponsesRowBase objects
        # Group by prompt index
        logger.debug("Organizing {} responses", len(responses))
        responses_by_index = {}
        for idx, text in zip(prompt_indices, response_texts, strict=True):
            if idx not in responses_by_index:
                responses_by_index[idx] = []
            responses_by_index[idx].append(text)

        response_rows = [
            ResponsesRowBase(index=idx, responses=texts)
            for idx, texts in responses_by_index.items()
        ]

        # Validate with ResponsesDatasetBase
        try:
            responses_dataset = ResponsesDatasetBase(responses=response_rows)
        except Exception as e:
            logger.error(
                "Failed to validate responses against ResponsesDatasetBase: {}", e
            )
            raise

        # Save results
        await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
        # Use experiment name and model ID as filename
        safe_model_id = model_to_use.replace(":", "_").replace("/", "_")
        output_path = output_dir / f"{config.experiment.name}_{safe_model_id}.parquet"

        # Convert back to polars for saving
        output_df = pl.DataFrame([r.model_dump() for r in responses_dataset.responses])

        await asyncio.to_thread(output_df.write_parquet, output_path)
        logger.info(
            "Inference complete for {}. Results saved to {}", model_to_use, output_path
        )


async def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run PGAIS inference.")
    parser = add_common_arguments(parser)
    args = parser.parse_args()

    # Configure logging to work with tqdm
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), level=args.log_level, colorize=True)

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path

    config = ConfigBase.from_toml(config_path)

    # Resolve paths
    input_path = Path(args.input) if args.input else config.data.input
    if not input_path.is_absolute():
        input_path = ROOT_DIR / input_path

    output_dir = Path(args.output) if args.output else config.output_dir

    await run_inference(
        config=config,
        input_path=input_path,
        output_dir=output_dir,
        model_id=args.model,
        temperature=args.temperature,
        max_concurrency=args.max_concurrency,
        rpm_limit=args.rpm_limit,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    asyncio.run(main())
