import gradio as gr
from chatbot_2 import Chatbot
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotUI:
    def __init__(self):
        logger.info("Initializing chatbot...")
        self.chatbot = Chatbot()
        logger.info("Chatbot ready!")

    def respond(self, message: str) -> str:
        """Generate response without history parameter"""
        try:
            response = self.chatbot.chat(message)
            logger.info(f"User: {message}")
            logger.info(f"Bot: {response}")
            return response
        except Exception as e:
            logger.error(f"Error in chat response: {e}")
            return "I apologize, but I encountered an error processing your request."

    def create_demo(self) -> gr.Blocks:
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            with gr.Column(elem_id="chat-container"):
                with gr.Column(elem_id="welcome-screen"):
                    gr.Markdown("# 🤖 DocAnalyzer AI Assistant", elem_classes=["main-title"])
                    gr.Markdown("Your intelligent companion for document analysis and general knowledge", elem_classes=["subtitle"])
                    
                    with gr.Row(elem_id="features-grid"):
                        with gr.Column(elem_classes=["feature-card"]):
                            gr.Markdown("📄 **Document Analysis**\nAnalyze any document for key insights and summaries")
                        with gr.Column(elem_classes=["feature-card"]):
                            gr.Markdown("❓ **Question Answering**\nGet accurate answers to your questions")
                        with gr.Column(elem_classes=["feature-card"]):
                            gr.Markdown("🔍 **Information Extraction**\nExtract specific information from documents")
                
                chatbox = gr.Chatbot(
                    value=[],
                    elem_id="chatbox",
                    height=450,
                    show_label=False,
                    container=True,
                )
                
                msg = gr.Textbox(
                    placeholder="Type your message here...",
                    container=True,
                    show_label=False,
                )

                with gr.Row():
                    submit = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")

                gr.Examples(
                    examples=[
                        "What is the capital of France?",
                        "Can you analyze this document for me?",
                        "Tell me about machine learning.",
                    ],
                    inputs=msg,
                )

                # Enhanced CSS for better styling
                gr.HTML("""
                <style>
                #chat-container {
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                    padding: 20px;
                    max-width: 800px;
                    margin: 0 auto;
                }
                .main-title {
                    font-size: 2.5em !important;
                    text-align: center;
                    margin-bottom: 0 !important;
                    color: var(--body-text-color) !important;
                }
                .subtitle {
                    text-align: center;
                    color: var(--body-text-color-subdued);
                    margin-bottom: 2em !important;
                }
                #features-grid {
                    margin-bottom: 2em;
                }
                .feature-card {
                    background: var(--background-fill-primary);
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                    transition: transform 0.2s;
                    border: 1px solid var(--border-color-primary);
                }
                .feature-card:hover {
                    transform: translateY(-5px);
                    background: var(--background-fill-secondary);
                }
                #chatbox {
                    border: 1px solid var(--border-color-primary);
                    border-radius: 8px;
                    background: var(--background-fill-primary);
                }
                #welcome-screen.hidden {
                    display: none;
                }
                </style>
                """)

            def bot_response(message, history):
                # Hide welcome screen on first message
                gr.HTML("""
                    <script>
                    document.getElementById('welcome-screen').classList.add('hidden');
                    </script>
                """)
                bot_message = self.respond(message)
                history.append((message, bot_message))
                return "", history

            msg.submit(bot_response, [msg, chatbox], [msg, chatbox])
            submit.click(bot_response, [msg, chatbox], [msg, chatbox])
            clear.click(lambda: None, None, chatbox)

        return demo

def main():
    ui = ChatbotUI()
    demo = ui.create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )

if __name__ == "__main__":
    main()
