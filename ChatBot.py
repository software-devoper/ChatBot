from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import streamlit as st
import json
import datetime

load_dotenv()

# Configure page
st.set_page_config(
    page_title="AI Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean chat UI
st.markdown("""
<style>
    /* Main chat area */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b2b2b;
        margin-left: 20%;
    }
    .chat-message.assistant {
        background-color: #1a1a1a;
        margin-right: 20%;
    }
    .message-content {
        line-height: 1.6;
    }
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #1a1a1a;
    }
    /* Input area */
    .input-area {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #0e0e0e;
        padding: 1rem;
        border-top: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content='You are a helpful AI assistant.')
    ]

if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None

if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = {}

if 'new_chat_counter' not in st.session_state:
    st.session_state.new_chat_counter = 0

def initialize_model():
    """Initialize the Gemini model"""
    return ChatGoogleGenerativeAI(
        model='gemini-2.5-flash',
        temperature=0.7
    )

def create_new_chat():
    """Create a new chat session"""
    st.session_state.new_chat_counter += 1
    chat_id = f"chat_{st.session_state.new_chat_counter}"
    st.session_state.chat_sessions[chat_id] = {
        "title": "New Chat",
        "messages": [SystemMessage(content='You are a helpful AI assistant.')],
        "created_at": datetime.datetime.now().isoformat()
    }
    st.session_state.current_chat_id = chat_id
    return chat_id

def get_current_chat():
    """Get current chat messages"""
    if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.chat_sessions:
        return st.session_state.chat_sessions[st.session_state.current_chat_id]["messages"]
    return [SystemMessage(content='You are a helpful AI assistant.')]

def update_current_chat(messages):
    """Update current chat messages"""
    if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.chat_sessions:
        st.session_state.chat_sessions[st.session_state.current_chat_id]["messages"] = messages
        # Update title based on first user message
        if len(messages) > 1 and isinstance(messages[1], HumanMessage):
            first_msg = messages[1].content[:30]
            st.session_state.chat_sessions[st.session_state.current_chat_id]["title"] = f"{first_msg}..."

def main():
    # Initialize model
    model = initialize_model()
    
    # Sidebar - Chat History
    with st.sidebar:
        st.title("💬 Chat History")
        
        # New Chat button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            create_new_chat()
            st.rerun()
        
        st.divider()
        
        # Chat sessions list
        if st.session_state.chat_sessions:
            for chat_id, chat_data in st.session_state.chat_sessions.items():
                is_selected = chat_id == st.session_state.current_chat_id
                button_label = f"💬 {chat_data['title']}"
                
                if st.button(button_label, 
                           key=chat_id, 
                           use_container_width=True,
                           type="primary" if is_selected else "secondary"):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()
        else:
            st.info("No chat history yet")
        
        st.divider()
        
        # Clear all chats
        if st.session_state.chat_sessions and st.button("🗑️ Clear All Chats", use_container_width=True):
            st.session_state.chat_sessions = {}
            st.session_state.current_chat_id = None
            st.rerun()

    # Main chat area
    st.title("🤖 AI Chat Assistant")
    
    # Get current chat messages
    current_messages = get_current_chat()
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in current_messages:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)
    
    # Chat input at bottom
    st.markdown("<br><br><br>", unsafe_allow_html=True)  # Add space for input
    
    # Use form for clean input handling
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        
        with col1:
            user_input = st.text_area(
                "Type your message...",
                placeholder="Send a message...",
                height=100,
                label_visibility="collapsed",
                key="user_input"
            )
        
        with col2:
            submit_button = st.form_submit_button(
                "Send",
                use_container_width=True,
                type="primary"
            )
    
    # Handle message submission
    if submit_button and user_input.strip():
        # Add user message to current chat
        current_messages.append(HumanMessage(content=user_input))
        
        # Display user message immediately
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
        
        # Get AI response
        with st.spinner("Thinking..."):
            try:
                response = model.invoke(current_messages)
                ai_response = response.content
                
                # Add AI response to chat
                current_messages.append(AIMessage(content=ai_response))
                
                # Update chat session
                update_current_chat(current_messages)
                
                # Display AI response
                with chat_container:
                    with st.chat_message("assistant"):
                        st.markdown(ai_response)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Initialize first chat if none exists
    if not st.session_state.current_chat_id and not st.session_state.chat_sessions:
        create_new_chat()

if __name__ == "__main__":
    main()