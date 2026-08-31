"""
Model factory — creates LangChain chat model instances.

Cloud-hosted models are accessed through OpenRouter.
Local models are accessed through Ollama.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from hklii_psla.config import settings


def _get_openrouter_model(model: str) -> BaseChatModel:
    """
    Create a LangChain chat model using OpenRouter.
    """
    from langchain_openrouter import ChatOpenRouter  # pyright: ignore[reportMissingImports]

    return ChatOpenRouter(
        model=model,
        api_key=settings.OPENROUTER_API_KEY,  # type: ignore[arg-type]
        temperature=settings.DEFAULT_TEMPERATURE,
        # max_tokens=settings.DEFAULT_MAX_TOKENS,
    )


def get_deepseek_model() -> BaseChatModel:
    return _get_openrouter_model(settings.OPENROUTER_DEEPSEEK_MODEL)


def get_qwen_model() -> BaseChatModel:
    return _get_openrouter_model(settings.OPENROUTER_QWEN_MODEL)


def get_glm_model() -> BaseChatModel:
    return _get_openrouter_model(settings.OPENROUTER_GLM_MODEL)


def get_kimi_model() -> BaseChatModel:
    return _get_openrouter_model(settings.OPENROUTER_KIMI_MODEL)


def get_minimax_model() -> BaseChatModel:
    return _get_openrouter_model(settings.OPENROUTER_MINIMAX_MODEL)


def get_ollama_model() -> BaseChatModel:
    from langchain_ollama import ChatOllama  # pyright: ignore[reportMissingImports]

    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.DEFAULT_TEMPERATURE,
    )

def get_gpt_model() -> BaseChatModel:
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
        api_key=settings.AZURE_OPENAI_API_KEY,  # type: ignore
        api_version=settings.AZURE_OPENAI_API_VERSION,
        max_retries=1,
    )


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_FACTORIES = {
    "deepseek": get_deepseek_model,
    "qwen": get_qwen_model,
    "glm": get_glm_model,
    "kimi": get_kimi_model,
    "minimax": get_minimax_model,
    "ollama": get_ollama_model,
    "gpt": get_gpt_model,
}

MODEL_PROVIDER_NAMES = {
    "gpt": settings.AZURE_OPENAI_DEPLOYMENT_NAME,
    # "deepseek": settings.OPENROUTER_DEEPSEEK_MODEL,
    # "qwen": settings.OPENROUTER_QWEN_MODEL,
    # "glm": settings.OPENROUTER_GLM_MODEL,
    # "kimi": settings.OPENROUTER_KIMI_MODEL,
    # "minimax": settings.OPENROUTER_MINIMAX_MODEL,
    # "ollama": settings.OLLAMA_MODEL,
}


def create_model(model_family: str) -> BaseChatModel:
    """
    Create a LangChain chat model.

    Parameters
    ----------
    model_family:
        One of:
            - deepseek
            - qwen
            - glm
            - kimi
            - minimax
            - ollama
    """
    factory = MODEL_FACTORIES.get(model_family)

    if factory is None:
        raise ValueError(
            f"Unknown model family: {model_family}. " +
            f"Available: {list(MODEL_FACTORIES.keys())}"
        )

    return factory()
