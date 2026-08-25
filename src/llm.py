"""Azure OpenAI integration. This is the only answer/summary generation path --
there is no offline/extractive fallback. The app requires AZURE_OPENAI_ENDPOINT,
OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT to be set (env vars or .env file).
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

_AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
_AZURE_API_KEY = os.environ.get("OPENAI_API_KEY")
_AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
_AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
_AZURE_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
_AZURE_EMBEDDING_API_VERSION = os.environ.get("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-02-01")

_REQUIRED = {
    "AZURE_OPENAI_ENDPOINT": _AZURE_ENDPOINT,
    "OPENAI_API_KEY": _AZURE_API_KEY,
    "AZURE_OPENAI_DEPLOYMENT": _AZURE_DEPLOYMENT,
}


def is_llm_available() -> bool:
    return all(_REQUIRED.values())


def chat_complete(system: str, user: str) -> str:
    """Calls Azure OpenAI. Raises RuntimeError/openai errors if misconfigured or the call fails."""
    missing = [name for name, value in _REQUIRED.items() if not value]
    if missing:
        raise RuntimeError(f"Azure OpenAI is not configured. Missing: {', '.join(missing)}")

    from openai import AzureOpenAI  # imported lazily so the package is optional

    client = AzureOpenAI(
        api_key=_AZURE_API_KEY,
        azure_endpoint=_AZURE_ENDPOINT,
        api_version=_AZURE_API_VERSION,
    )
    response = client.chat.completions.create(
        model=_AZURE_DEPLOYMENT,  # Azure calls models by deployment name, not model name
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeds a batch of texts via the Azure OpenAI embeddings deployment
    (AZURE_OPENAI_EMBEDDING_DEPLOYMENT, default "text-embedding-3-small").
    """
    missing = [name for name, value in _REQUIRED.items() if not value]
    if missing:
        raise RuntimeError(f"Azure OpenAI is not configured. Missing: {', '.join(missing)}")
    if not texts:
        return []

    from openai import AzureOpenAI  # imported lazily so the package is optional

    client = AzureOpenAI(
        api_key=_AZURE_API_KEY,
        azure_endpoint=_AZURE_ENDPOINT,
        api_version=_AZURE_EMBEDDING_API_VERSION,
    )
    response = client.embeddings.create(model=_AZURE_EMBEDDING_DEPLOYMENT, input=texts)
    return [item.embedding for item in response.data]
