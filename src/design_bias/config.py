"""Configuration and data schemas for design_bias."""

from pydantic import Field

from prismal.data import PromptDatasetBase, PromptRowBase


class DesignPromptRow(PromptRowBase):
    """Custom prompt row for design_bias experiments.

    Attributes:
        framework: The web framework used (e.g., bootstrap, tailwind).
        descriptor: The descriptor for the design.
        system_prompt: The system prompt used for generation.
        user_prompt: The user prompt used for generation.
    """

    framework: str | None = Field(default=None, description="The web framework used.")
    descriptor: str | None = Field(
        default=None, description="The descriptor for the design."
    )
    system_prompt: str = Field(..., description="The system prompt used.")
    user_prompt: str = Field(..., description="The user prompt used.")


class DesignPromptDataset(PromptDatasetBase):
    """Custom prompt dataset for design_bias experiments.

    Attributes:
        prompts: A list of DesignPromptRow objects.
    """

    prompts: list[DesignPromptRow]
