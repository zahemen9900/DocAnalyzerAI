import streamlit as st
from src.experiments.chatbot_2 import FinancialChatbot
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamlitUI:
    def __init__(self):
        self.chatbot = FinancialChatbot()
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
    def create_ui(self):
        # Set page config
        st.set_page_config(
            page_title="DocAnalyzer AI Assistant",
            page_icon="🤖",
            layout="centered"
        )
        
        # Header
        st.title("🤖 DocAnalyzer AI Assistant")
        st.markdown("Your intelligent companion for document analysis")
        
        # Sidebar with clear button
        with st.sidebar:
            if st.button("Clear Chat", key="clear"):
                st.session_state.messages = []
                st.rerun()  # Use st.rerun() instead of experimental_rerun()
        
        # Features section (visible only when chat is empty)
        if not st.session_state.messages:
            with st.container():
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 📄 Document Analysis")
                    st.write("Analyze documents for key insights")
                    
                with col2:
                    st.markdown("### ❓ Question Answering")
                    st.write("Get accurate answers to queries")
                    
                with col3:
                    st.markdown("### 🔍 Information Extraction")
                    st.write("Extract specific information")
        
        # Chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get bot response
            with st.chat_message("assistant"):
                response = self.chatbot.chat(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    ui = StreamlitUI()
    ui.create_ui()

if __name__ == "__main__":
    main()
