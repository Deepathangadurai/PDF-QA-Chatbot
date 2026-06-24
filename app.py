import streamlit as str_layout
import streamlit as st
from utils import (
    extract_text_from_pdfs, 
    get_text_chunks, 
    create_vector_store, 
    generate_gemini_response
)

# 1. Page Configuration
st.set_page_config(
    page_title="PDF QA Chatbot",
    page_icon="📚",
    layout="wide"
)

# 2. Session State Initialization
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# 3. Sidebar Layout for PDF Uploads
with st.sidebar:
    st.title("📁 Document Control Panel")
    st.markdown("Upload your PDF reference materials here to initialize the knowledge engine.")
    
    uploaded_files = st.file_uploader(
        "Select PDF Files", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    process_button = st.button("Process Documents", use_container_width=True)
    
    if process_button:
        if not uploaded_files:
            st.error("Please upload at least one PDF file before clicking process.")
        else:
            with st.spinner("Extracting text and indexing embeddings..."):
                try:
                    # Pipeline Execution
                    raw_text = extract_text_from_pdfs(uploaded_files)
                    
                    if not raw_text.strip():
                        st.error("No extractable text found within the provided PDFs.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        vector_db = create_vector_store(text_chunks)
                        
                        # Save back to global state
                        st.session_state.vector_store = vector_db
                        st.success(f"Successfully processed {len(uploaded_files)} document(s)!")
                except Exception as e:
                    st.error(f"An unexpected error occurred during setup: {e}")

# 4. Main Chat Interface Presentation
st.title("📚 Contextual PDF QA Assistant")
st.markdown("Ask natural language questions grounded specifically within your uploaded reference assets.")

# Display existing chat timeline
for interaction in st.session_state.conversation_history:
    with st.chat_message(interaction["role"]):
        st.markdown(interaction["content"])

# User Input Entry Control
if user_query := st.chat_input("Ask something about your documents..."):
    # Render user query instantly
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.conversation_history.append({"role": "user", "content": user_query})

    # Validate vector pipeline status
    if st.session_state.vector_store is None:
        with st.chat_message("assistant"):
            warning_msg = "Please upload and process PDF documents via the sidebar before asking questions."
            st.warning(warning_msg)
        st.session_state.conversation_history.append({"role": "assistant", "content": warning_msg})
    else:
        # Generate Response Block
        with st.chat_message("assistant"):
            with st.spinner("Scanning index and thinking..."):
                try:
                    # Perform similarity search fetching top 4 text nodes
                    matching_docs = st.session_state.vector_store.similarity_search(user_query, k=4)
                    compiled_context = "\n\n".join([doc.page_content for doc in matching_docs])
                    
                    # Generate Answer
                    bot_reply = generate_gemini_response(compiled_context, user_query)
                    st.markdown(bot_reply)
                    
                    # Log to application session timeline
                    st.session_state.conversation_history.append({"role": "assistant", "content": bot_reply})
                except Exception as e:
                    error_msg = f"Failed to retrieve or generate answer: {str(e)}"
                    st.error(error_msg)
                    st.session_state.conversation_history.append({"role": "assistant", "content": error_msg})