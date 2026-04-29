"""src/interface.py
Gradio UI for the Financial Q&A agent.
"""

import inspect
import gradio as gr


def build_gradio_app(agent) -> gr.Blocks:
    with gr.Blocks(title="Financial Q&A Agent") as app:

        gr.Markdown(
            "# Financial Q&A Agent\n"
            "Ask questions about financial reports (10-K, 10-Q, earnings)."
        )

        with gr.Row():

            with gr.Column(scale=3):
                chatbot_params = inspect.signature(gr.Chatbot.__init__).parameters
                chatbot_kwargs = {"label": "Conversation", "height": 520}
                if "type" in chatbot_params:
                    chatbot_kwargs["type"] = "tuples"
                if "show_copy_button" in chatbot_params:
                    chatbot_kwargs["show_copy_button"] = True

                chatbot = gr.Chatbot(**chatbot_kwargs)
                use_tuple = chatbot_kwargs.get("type") == "tuples"

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder='e.g. "What is Apple\'s FY2022 net income?"',
                        label="Your question",
                        scale=4,
                        lines=2,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("Clear session", variant="secondary")
                    memory_btn = gr.Button("Show memory", variant="secondary")

                gr.Examples(
                    examples=[
                        "What is Apple's FY2022 net income?",
                        "What was Amazon's revenue in Q1 2023?",
                        "What are the main risk factors cited by Boeing in their most recent 10-K?",
                        "And how does that compare to the previous year?",
                        "Show memory",
                    ],
                    inputs=msg_input,
                    label="Examples",
                )

            with gr.Column(scale=1):
                gr.Markdown("### Session state")
                status_box = gr.Textbox(
                    label="Memory",
                    value="Agent not initialized yet.",
                    lines=8,
                    interactive=False,
                )
                gr.Markdown(
                    "**Short-term:** last 6 turns\n\n"
                    "**Mid-term:** summary every 6 turns\n\n"
                    "**Long-term:** memory/long_term.json"
                )

        def _append_turn(history, user_text, assistant_text):
            base = list(history or [])
            if use_tuple:
                base.append([user_text, assistant_text])
            else:
                base.append({"role": "user", "content": user_text})
                base.append({"role": "assistant", "content": assistant_text})
            return base

        def chat_fn(message, history):
            if not message.strip():
                return history, "", agent._get_status_text()
            response = agent.answer_question(message, history)
            return _append_turn(history, message, response), "", agent._get_status_text()

        def show_memory(history):
            report = agent._format_memory_report() if agent._long_term_mem else "Agent not initialized yet."
            return _append_turn(history, "Show memory", report), agent._get_status_text()

        send_btn.click(
            fn=chat_fn,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, status_box],
        )
        msg_input.submit(
            fn=chat_fn,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, status_box],
        )
        clear_btn.click(
            fn=agent.clear_session,
            outputs=[chatbot, status_box],
        )
        memory_btn.click(
            fn=show_memory,
            inputs=[chatbot],
            outputs=[chatbot, status_box],
        )

        app.unload(agent.on_session_end)

    return app
