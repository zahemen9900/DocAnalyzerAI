import gradio as gr

# Simple Chat Function
def chat_response(message):
    return f"You said: {message}"

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("### Chat Interface with Custom Send Icon Button")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Type your message here")
    send_btn = gr.Button("", elem_id="send-button")  # Empty button for customization
    
    send_btn.click(chat_response, inputs=msg, outputs=chatbot)
    
    # Custom HTML for FontAwesome Icon
    gr.HTML("""
    <style>
        #send-button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 50%;
            padding: 10px;
            cursor: pointer;
            font-size: 20px;
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        #send-button:hover {
            background-color: #45a049;
        }

        #send-button i {
            font-size: 20px;
            color: white;
        }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const sendButton = document.getElementById('send-button');
            if (sendButton) {
                sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
            }
        });
    </script>
    """)

demo.launch()
