import logging

from rag.llm_config import LLM_PROVIDER
from rag.src.llm_ollama import OllamaLLM
from rag.src.llm_anthropic import AnthropicLLM

logger = logging.getLogger(__name__)


def get_llm():
    if LLM_PROVIDER == "anthropic":
        logger.info("LLM provider: anthropic")
        return AnthropicLLM()
    logger.info("LLM provider: ollama")
    return OllamaLLM()
