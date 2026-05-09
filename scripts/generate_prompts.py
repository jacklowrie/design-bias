"""Script to generate prompts for design_bias experiments.

This script takes descriptors, frameworks, system prompt template, and user prompt
template as input and generates all possible combinations of them. The output is a
parquet file containing a DesignPromptDataset-compliant set of prompts.
"""

import argparse
from pathlib import Path

import polars as pl
from loguru import logger
from tqdm import tqdm

from design_bias.config import DesignPromptDataset, DesignPromptRow
from prismal.config import DATA_DIR


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate prompts for design_bias.")
    parser.add_argument(
        "-d",
        "--descriptors",
        help="Path to the descriptors file relative to data/.",
    )
    parser.add_argument(
        "-f",
        "--frameworks",
        help="Path to the frameworks file relative to data/.",
    )
    parser.add_argument(
        "-s",
        "--system-prompt-template",
        required=True,
        help="Path to the system prompt template file relative to data/.",
    )
    parser.add_argument(
        "-u",
        "--user-prompt-template",
        required=True,
        help="Path to the user prompt template file relative to data/.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        default="outputs/prompts.parquet",
        help="Path to the output parquet file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_args()

    # Configure logging to work with tqdm
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), level=args.log_level, colorize=True)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read inputs
    if args.descriptors:
        descriptors_path = DATA_DIR / args.descriptors
        with descriptors_path.open("r", encoding="utf-8") as f:
            descriptors: list[str | None] = [
                line.strip() if line.strip() != "None" else None
                for line in f
                if line.strip()
            ]
    else:
        descriptors = [None]

    if not descriptors:
        descriptors = [None]

    if args.frameworks:
        frameworks_path = DATA_DIR / args.frameworks
        with frameworks_path.open("r", encoding="utf-8") as f:
            frameworks: list[str | None] = [
                line.strip() if line.strip() != "None" else None
                for line in f
                if line.strip()
            ]
    else:
        frameworks = [None]

    if not frameworks:
        frameworks = [None]

    system_prompt_path = DATA_DIR / args.system_prompt_template
    user_prompt_path = DATA_DIR / args.user_prompt_template

    with system_prompt_path.open("r", encoding="utf-8") as f:
        system_prompt_template = f.read().strip()

    with user_prompt_path.open("r", encoding="utf-8") as f:
        user_prompt_template = f.read().strip()

    # Generate combinations
    prompts = []
    idx = 0
    for descriptor in descriptors:
        for framework in frameworks:
            system_prompt = system_prompt_template

            # Apply conditional formatting to user prompt
            user_prompt = user_prompt_template

            if framework:
                user_prompt = user_prompt.replace("{framework}", framework)
            else:
                # Remove " built with {framework}" or "built with {framework}"
                # The template has "built with {framework}."
                user_prompt = user_prompt.replace(" built with {framework}", "")
                user_prompt = user_prompt.replace("built with {framework}", "")

            if descriptor:
                user_prompt = user_prompt.replace("{descriptor}", descriptor)
            else:
                # Remove the sentence: "The page should have a {descriptor} design."
                # We'll look for the specific line if possible, or use a more general
                # regex/replacement.
                # In userprompt_template.txt it is the last line.
                target = "The page should have a {descriptor} design."
                if target in user_prompt:
                    user_prompt = user_prompt.replace(target, "")
                else:
                    # Fallback for slight variations
                    import re

                    user_prompt = re.sub(
                        r"[^\n]*\{descriptor\}[^\n]*\n?", "", user_prompt
                    )

            user_prompt = user_prompt.strip()

            # DesignPromptRow inherits from PromptRowBase which needs 'prompt'
            # and 'index'. It also has framework, descriptor, system_prompt,
            # and user_prompt.
            row = DesignPromptRow(
                index=idx,
                prompt=user_prompt,  # Using user_prompt as the main 'prompt'
                framework=framework,
                descriptor=descriptor,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            prompts.append(row)
            idx += 1

    # Validate with DesignPromptDataset
    dataset = DesignPromptDataset(prompts=prompts)

    # Convert to polars DataFrame and save to parquet
    # We use model_dump() to get dictionaries
    df = pl.DataFrame([p.model_dump() for p in dataset.prompts])
    df.write_parquet(output_path)

    logger.info("Generated {} prompts and saved to {}", len(prompts), output_path)


if __name__ == "__main__":
    main()
