import gradio as gr
from rag.doc_chunk import rag_query_answer

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
            if not user_input.strip():
                return "", chat_history

            bot_response = generate_response(user_input, chat_history)

            chat_history = chat_history or []
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": bot_response})

            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        submit.click(respond, [msg, chatbot], [msg, chatbot])

    return interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
    )




def legal_aid_chat_interface2():
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
            if not user_input.strip():
                return "", chat_history

            bot_response = rag_query_answer(user_input, chat_history)

            chat_history = chat_history or []
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": bot_response})

            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        submit.click(respond, [msg, chatbot], [msg, chatbot])

    return interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
    )