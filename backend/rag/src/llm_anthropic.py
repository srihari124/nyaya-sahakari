import logging
import time

from rag.llm_config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    ANTHROPIC_TIMEOUT_SEC,
    ANTHROPIC_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


class AnthropicLLM:
    def __init__(self):
        import anthropic
        self.model = ANTHROPIC_MODEL
        self.max_tokens = ANTHROPIC_MAX_TOKENS
        self.client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            timeout=float(ANTHROPIC_TIMEOUT_SEC),
        )
        logger.info("AnthropicLLM initialized: model=%s", self.model)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            start = time.perf_counter()
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
            )
            elapsed = time.perf_counter() - start
            logger.info("Anthropic generation completed in %.3fs", elapsed)
            return message.content[0].text.strip()
        except Exception as e:
            logger.error("Anthropic generation error: %s", e)
            return f"LLM Error: {str(e)}"
