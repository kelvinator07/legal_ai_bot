import gradio as gr

from legal_agents.runner import run_legal_agent_sync

MAX_USER_MESSAGE_CHARS = 32_000
_USER_MESSAGE_PREVIEW_CHARS = 400

CUSTOM_CSS = """
/* ── Layout ── */
.gradio-container {
    max-width: 860px !important;
    margin: 0 auto !important;
    padding: 0 16px !important;
}

/* ── Header ── */
#header-bar {
    background: linear-gradient(135deg, #0f3d1e 0%, #1a6b3c 50%, #22915a 100%);
    border-radius: 16px;
    padding: 28px 32px 22px;
    margin-bottom: 12px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
#header-bar::before {
    content: "";
    position: absolute;
    top: -40%;
    right: -10%;
    width: 260px;
    height: 260px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
#header-bar h1 {
    color: #fff !important;
    margin: 0 0 2px !important;
    font-size: 1.6em !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px;
}
#header-bar .subtitle {
    color: #b8e6cc !important;
    font-size: 0.88em;
    margin: 0;
}
#header-bar .badge-row {
    display: flex;
    justify-content: center;
    gap: 6px;
    flex-wrap: wrap;
    margin-top: 14px;
}
#header-bar .badge {
    background: rgba(255,255,255,0.13);
    color: #d4f5e2;
    padding: 3px 11px;
    border-radius: 20px;
    font-size: 0.73em;
    font-weight: 500;
    backdrop-filter: blur(4px);
    border: 1px solid rgba(255,255,255,0.1);
}

/* ── Chat bubbles ── */
#chatbot {
    border-radius: 14px !important;
    border: 1px solid var(--border-color-primary) !important;
}
#chatbot .message {
    max-width: 85% !important;
    font-size: 0.92em !important;
    line-height: 1.55 !important;
}
#chatbot .bot {
    border-radius: 14px 14px 14px 4px !important;
}
#chatbot .user {
    border-radius: 14px 14px 4px 14px !important;
}

/* ── Input area ── */
#input-row {
    border: 2px solid var(--border-color-primary);
    border-radius: 14px;
    padding: 4px;
    background: var(--background-fill-primary);
    transition: border-color 0.2s;
}
#input-row:focus-within {
    border-color: #22915a;
}
#msg-box textarea {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    font-size: 0.93em !important;
    padding: 10px 14px !important;
}
#send-btn {
    background: linear-gradient(135deg, #1a6b3c, #22915a) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    min-height: 44px;
    font-weight: 600 !important;
    font-size: 0.9em !important;
    transition: opacity 0.15s;
    cursor: pointer;
}
#send-btn:hover {
    opacity: 0.88;
}

/* ── Example buttons ── */
.example-btn {
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 10px !important;
    background: var(--background-fill-secondary) !important;
    font-size: 0.82em !important;
    padding: 8px 14px !important;
    transition: all 0.15s !important;
    cursor: pointer !important;
}
.example-btn:hover {
    border-color: #22915a !important;
    background: var(--background-fill-primary) !important;
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

/* ── Disclaimer ── */
#disclaimer-text {
    font-size: 0.78em !important;
    opacity: 0.6;
    text-align: center;
    margin-top: 4px !important;
}

/* ── Action buttons ── */
.action-btn {
    border-radius: 10px !important;
    font-size: 0.85em !important;
    min-height: 44px;
}

/* ── Footer ── */
footer { display: none !important; }
"""


def legal_aid_chat_interface():
    with gr.Blocks(
        title="Pan-African Legal Aid Chatbot",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.green,
            secondary_hue=gr.themes.colors.emerald,
            neutral_hue=gr.themes.colors.gray,
            font=gr.themes.GoogleFont("Inter"),
        ),
    ) as interface:

        # ── Header ──
        gr.HTML(
            """
            <div id="header-bar">
                <h1>Pan-African Legal Aid</h1>
                <p class="subtitle">Free legal guidance grounded in real Nigerian statutes</p>
                <div class="badge-row">
                    <span class="badge">Labour</span>
                    <span class="badge">Tenancy</span>
                    <span class="badge">Consumer Rights</span>
                    <span class="badge">Contracts</span>
                    <span class="badge">Constitution</span>
                </div>
            </div>
            """
        )

        # ── Chat ──
        chatbot = gr.Chatbot(
            label="Legal Advisor",
            height=400,
            elem_id="chatbot",
            placeholder="Ask a legal question to get started...",
            layout="bubble",
            show_label=False,
            avatar_images=(None, "https://em-content.zobj.net/source/twitter/408/balance-scale_2696-fe0f.png"),
        )

        # ── Input row ──
        with gr.Group(elem_id="input-row"):
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about your legal rights...",
                    lines=1,
                    max_lines=4,
                    scale=6,
                    show_label=False,
                    container=False,
                    elem_id="msg-box",
                )
                submit = gr.Button("Send", elem_id="send-btn", scale=1, min_width=80)

        # ── Examples ──
        with gr.Accordion("Example questions", open=False):
            with gr.Row():
                ex1 = gr.Button("My boss deducted my salary for being late", elem_classes="example-btn", size="sm")
                ex2 = gr.Button("Can my landlord evict me without notice?", elem_classes="example-btn", size="sm")
            with gr.Row():
                ex3 = gr.Button("I bought a defective phone — what are my rights?", elem_classes="example-btn", size="sm")
                ex4 = gr.Button("I was fired without warning — what can I do?", elem_classes="example-btn", size="sm")
            with gr.Row():
                ex5 = gr.Button("Review this contract for risky clauses", elem_classes="example-btn", size="sm")
                ex6 = gr.Button("How do I file a complaint against my employer?", elem_classes="example-btn", size="sm")

        # ── Actions row ──
        clear = gr.ClearButton([msg, chatbot], value="New conversation", elem_classes="action-btn")

        # ── Disclaimer ──
        gr.Markdown(
            "This chatbot provides general legal information, not legal advice. "
            "For serious matters, consult a qualified lawyer.",
            elem_id="disclaimer-text",
        )

        # Hidden state to pass the user text between chained steps
        pending_text = gr.State("")

        # ── Chat logic ──
        def set_cancel():
            return gr.Button("Cancel", elem_id="send-btn", variant="stop")

        def set_send():
            return gr.Button("Send", elem_id="send-btn", variant="primary")

        def show_user_message(user_input, chat_history):
            """Step 1: Show user message instantly, clear input, save text."""
            chat_history = chat_history or []
            raw = user_input if isinstance(user_input, str) else ""
            text = raw.replace("\x00", "").strip()
            if not text:
                return "", chat_history, ""
            chat_history.append({"role": "user", "content": text})
            return "", chat_history, text

        def get_response(saved_text, chat_history):
            """Step 2: Call the LLM using the saved text (same logic as original respond)."""
            chat_history = chat_history or []
            if not saved_text:
                return chat_history

            if len(saved_text) > MAX_USER_MESSAGE_CHARS:
                chat_history.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"Your message is too long ({len(saved_text):,} characters). "
                            f"I can accept up to {MAX_USER_MESSAGE_CHARS:,} characters. "
                            "Please shorten it or split it into smaller messages."
                        ),
                    },
                )
                return chat_history

            # Pass history WITHOUT the user message we just added,
            # because run_legal_agent_sync adds it again internally
            history_for_agent = chat_history[:-1]

            try:
                bot_response = run_legal_agent_sync(saved_text, history_for_agent)
            except Exception:
                bot_response = (
                    "Something went wrong while processing your request. "
                    "Check OPENROUTER_API_KEY / OPENAI_API_KEY, network access, and that "
                    "dependencies are installed. If it keeps happening, try again later."
                )

            chat_history.append({"role": "assistant", "content": bot_response})
            return chat_history

        def show_example(question, chat_history):
            """Step 1 for examples: Show question instantly."""
            chat_history = chat_history or []
            chat_history.append({"role": "user", "content": question})
            return "", chat_history, question

        # ── Wire: show message → cancel → LLM → send ──
        def wire_send(trigger_event, inputs):
            trigger_event(
                show_user_message, inputs=inputs, outputs=[msg, chatbot, pending_text]
            ).then(
                set_cancel, outputs=[submit]
            ).then(
                get_response, inputs=[pending_text, chatbot], outputs=[chatbot]
            ).then(
                set_send, outputs=[submit]
            )

        def wire_example(btn, question):
            btn.click(
                lambda h: show_example(question, h), inputs=[chatbot], outputs=[msg, chatbot, pending_text]
            ).then(
                set_cancel, outputs=[submit]
            ).then(
                get_response, inputs=[pending_text, chatbot], outputs=[chatbot]
            ).then(
                set_send, outputs=[submit]
            )

        wire_send(submit.click, [msg, chatbot])
        wire_send(msg.submit, [msg, chatbot])

        wire_example(ex1, "My boss deducted my salary for being late")
        wire_example(ex2, "Can my landlord evict me without notice?")
        wire_example(ex3, "I bought a defective phone — what are my rights?")
        wire_example(ex4, "I was fired without warning — what can I do?")
        wire_example(ex5, "Review this contract for risky clauses")
        wire_example(ex6, "How do I file a complaint against my employer?")

    return interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
    )
