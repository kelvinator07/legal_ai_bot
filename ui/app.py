import gradio as gr

from legal_agents.runner import run_legal_agent_sync

# Guard rails: keep UI entry bounded and avoid leaking internal errors to users.
MAX_USER_MESSAGE_CHARS = 32_000
_USER_MESSAGE_PREVIEW_CHARS = 400

LEGAL_KNOWLEDGE_BASE = {}


def generate_response(user_query, chat_history):
    """Generate a legal aid response based on user query."""
    return "This is a placeholder legal response."


def legal_aid_chat_interface():
    with gr.Blocks(title="Pan-African Legal Aid Chatbot") as interface:

        gr.Markdown("# ⚖️ Pan-African Legal Aid Chatbot")

        chatbot = gr.Chatbot(label="Legal Advisor Chat", height=500)

        with gr.Row():
            msg = gr.Textbox(
                label="Your Question",
                placeholder="Ask me about tenant rights, evictions, employment, property, or any legal matter...",
                lines=2,
            )
            submit = gr.Button("Send")

        gr.Markdown("### Example Questions")
        with gr.Row():
            gr.Button("What are my tenant rights?")
            gr.Button("Can my landlord evict me?")
            gr.Button("What should a lease include?")

        gr.Markdown(
            """
        **DISCLAIMER:** This is general information, not legal advice. 
        For serious legal matters, consult a qualified lawyer. Laws vary by country.
        """
        )

        def respond(user_input, chat_history):
            chat_history = chat_history or []
            raw = user_input if isinstance(user_input, str) else ""
            text = raw.replace("\x00", "").strip()
            if not text:
                return "", chat_history

            if len(text) > MAX_USER_MESSAGE_CHARS:
                preview = text[:_USER_MESSAGE_PREVIEW_CHARS]
                if len(text) > _USER_MESSAGE_PREVIEW_CHARS:
                    preview = preview + "…"
                chat_history.append({"role": "user", "content": preview})
                chat_history.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"Your message is too long ({len(text):,} characters). "
                            f"I can accept up to {MAX_USER_MESSAGE_CHARS:,} characters. "
                            "Please shorten it or split it into smaller messages."
                        ),
                    },
                )
                return "", chat_history

            try:
                bot_response = run_legal_agent_sync(text, chat_history)
            except Exception:
                bot_response = (
                    "Something went wrong while processing your request. "
                    "Check OPENROUTER_API_KEY / OPENAI_API_KEY, network access, and that "
                    "dependencies are installed. If it keeps happening, try again later."
                )

            chat_history.append({"role": "user", "content": text})
            chat_history.append({"role": "assistant", "content": bot_response})

            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        submit.click(respond, [msg, chatbot], [msg, chatbot])

    return interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
    )