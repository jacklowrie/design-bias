"""Inference clients for PGAIS."""

import os

from prismal.inference import AsyncOpenAIInferenceClient, OpenAIInferenceClient


class PGAISInferenceClient(OpenAIInferenceClient):
    """Specialized client for Purdue GenAI Studio."""

    DEFAULT_BASE_URL = "https://genai.rcac.purdue.edu/api"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 5,
    ) -> None:
        """Initialize the PGAIS client."""
        api_key = api_key or os.getenv("PGAIS_KEY")
        base_url = base_url or self.DEFAULT_BASE_URL
        super().__init__(api_key=api_key, base_url=base_url, max_retries=max_retries)


class AsyncPGAISInferenceClient(AsyncOpenAIInferenceClient):
    """Specialized async client for Purdue GenAI Studio."""

    DEFAULT_BASE_URL = "https://genai.rcac.purdue.edu/api"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 5,
        requests_per_minute: int = 20,
    ) -> None:
        """Initialize the async PGAIS client."""
        api_key = api_key or os.getenv("PGAIS_KEY")
        base_url = base_url or self.DEFAULT_BASE_URL
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            requests_per_minute=requests_per_minute,
        )
