import logging

from ui.app import legal_aid_chat_interface

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Gradio UI → Agent layer (OpenAI Agents SDK) → RAG + MCP legal tools + web search."""
    logger.info("[APP] Starting Legal AI Bot")
    legal_aid_chat_interface()


if __name__ == "__main__":
    main()
