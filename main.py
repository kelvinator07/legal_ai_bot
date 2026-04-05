from ui.app import legal_aid_chat_interface


def main() -> None:
    """Gradio UI → Agent layer (OpenAI Agents SDK) → RAG + MCP legal tools + web search."""
    print("Hello from legal-ai-bot!")
    legal_aid_chat_interface()


if __name__ == "__main__":
    main()
