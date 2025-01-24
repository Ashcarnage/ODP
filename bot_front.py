import streamlit as st
import requests
import tempfile
import os 
import io
import PyPDF2
import docx2txt
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from model import QueryModel,HisModel
import json
import base64
import clipboard
from PIL import Image
from shutil import copyfile
import uuid
url = "http://0.0.0.0:8000"
endpoint = "/Agent_Response"

def upload_image_to_server(image,max_size = (200, 200),quality=65):
    # unique_id = str(uuid.uuid4())
    # # image_bytes = image_file.getvalue()
    # # base64_image = base64.b64encode(image_bytes).decode()
    # image_url = f"https://yourserver.com/img/{unique_id}"
    # # image_url = f"data:image/{image_file.type.split('/')};base64,{base64_image}"
    # return image_url
   # Generate a unique filename for saving
    # unique_id = str(uuid.uuid4())
    # file_extension = os.path.splitext(image_file.name)[-1]  # Extract file extension
    # filename = f"{unique_id}"
    
    # # Save the file locally or on a server (example: local directory 'uploads/')
    # save_path = os.path.join(filename)
    # with open(save_path, "wb") as f:
    #     f.write(image_file.getvalue())  # Save file content
    
    # # Generate a public URL for accessing this file (e.g., hosted on your server)
    # image_url = f"https://yourserver.com/{filename}"
        # Generate a unique filename using UUID
    # unique_id = str(uuid.uuid4())
    # # file_extension = os.path.splitext(image_path)[-1]
    # file_extension = '.png'
    # unique_filename = f"{unique_id}{file_extension}"
    
    # # Create a temporary directory for serving files
    # serving_directory = os.path.join(os.getcwd(), "temp_images")
    # os.makedirs(serving_directory, exist_ok=True)
    
    # # Copy the file to the serving directory
    # # destinatio  
    # # Generate a relative path for the image
    # image_url = f"{serving_directory}/{unique_filename}"
    # return image_url
    
    # return image_url
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    # image.save(buffer, format="JPEG", quality=quality)
    # buffer.seek(0)
    # img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    buffered = io.BytesIO()
    
    # Save the image in the desired format with compression quality
    image.save(buffered, format='JPEG', quality=quality)
    
    # Get the base64 encoding of the compressed image
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return image_base64


def process_image(image,image_file):
    """Process uploaded image file"""
    try:
        # Create a temporary file with a proper extension
        img_base64 = upload_image_to_server(image)
        print(f"Uploaded image : {img_base64}")
        # Create a Document object with correct metadata
        doc = Document(
            page_content=f"[Image File: {image_file.name}]",
            metadata={
                "image_file":img_base64,
                "source": image_file.name,
                "type": "image",
                # "image_base64": img_base64,  # Ensure this path is correct
                "original_name": image_file.name
            }
        )
        return doc
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        print(f"Image processing error: {str(e)}")
        return None
def process_text(text_file):
    text_content = text_file.getvalue().decode('utf-8')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.create_documents([text_content])
    texts = text_splitter.split_documents(texts)
    return texts
def process_docx(docx_file):
    """Process uploaded Word document"""
    text_content = docx2txt.process(io.BytesIO(docx_file.getvalue()))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.create_documents([text_content])
    # print("===== PRE TEXTS : ",texts)
    texts = text_splitter.split_documents(texts)
    # print("--- DOCS GOT SPLIT ---")
    # print(texts)
    return texts

def process_pdf(pdf_file):
    """Process uploaded PDF file"""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.getvalue()))
    text_content = ""
    for page in pdf_reader.pages:
        text_content += page.extract_text()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.create_documents([text_content])
    texts = text_splitter.split_documents(texts)
    # print(texts)
    return texts

def get_response(query: QueryModel,chat_history: HisModel,documents = None):
    try:
        payload = {
            "query": query.model_dump(),
            "chat_history": chat_history.model_dump(),     
        }
        if documents:
            docs_dict = [{
                "page_content": doc.page_content,
                "metadata": doc.metadata
            } for doc in documents]
            # print("Processed documents:", docs_dict)
            payload["documents"] = docs_dict
            
        print("Final payload:", json.dumps(payload, indent=2))
        response = requests.post(url+endpoint,json=payload)
        # Print response details for debugging
        print("Response status:", response.status_code)
        print("Response content:", response.text[-1])
        
        if response.status_code != 200:
            error_detail = response.json().get('detail', 'Unknown error')
            return f"Error: {error_detail}"
        return response.json()["response"]
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        return f"Error: {str(e)}"



# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'user_query' not in st.session_state:
    st.session_state.user_query = ""
if 'submit_clicked' not in st.session_state:
    st.session_state.submit_clicked = False

def clear_input():
    """Callback to clear the input text"""
    st.session_state.user_query = ""
    st.session_state.submit_clicked = False

# Function to handle query submission
def handle_submit():
    st.session_state.submit_clicked = True

st.title("🤖 Intelligence with LLMs")

# File upload section
st.sidebar.header("Upload Files")
if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []

uploaded_files = st.sidebar.file_uploader(
    "Upload images or documents", 
    type=["png", "jpg", "jpeg", "pdf", "txt", "docx"],
    accept_multiple_files=True
)
documents = []
if uploaded_files:
    for file in uploaded_files:
        try:
            if file.name not in [doc.metadata.get("source") for doc in st.session_state.uploaded_documents]:
                if file.type.startswith('image'):
                    image = Image.open(file)
                    doc = process_image(image,file)
                    if doc:
                        documents.append(doc)
                        st.session_state.documents.append(doc)
                        st.session_state.uploaded_documents.append(doc)
                        st.sidebar.image(file, caption=file.name, use_container_width=True)
                
                elif file.type == 'text/plain':
                    docs = process_text(file)
                    documents.extend(docs)
                    st.session_state.documents.extend(docs)
                    st.session_state.uploaded_documents.extend(docs)
                    st.sidebar.success(f"Processed text file: {file.name}")
                
                elif file.type == 'application/pdf':
                    docs = process_pdf(file)
                    documents.extend(docs)
                    st.session_state.documents.extend(docs)
                    st.session_state.uploaded_documents.extend(docs)
                    st.sidebar.success(f"Processed PDF: {file.name}")
                
                elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                    docs = process_docx(file)
                    documents.extend(docs)
                    st.session_state.documents.extend(docs)
                    st.session_state.uploaded_documents.extend(docs)
                    st.sidebar.success(f"Processed Word document: {file.name}")
                    
        except Exception as e:
            st.sidebar.error(f"Error processing {file.name}: {str(e)}")

st.sidebar.header("Paste Screenshots")
if st.sidebar.button("Paste Screenshot"):
    try:
        image= clipboard.paste()
        if isinstance(image,Image.Image):
            with tempfile.NamedTemporaryFile(delete=False,suffix=".png") as f:
                image.save(f.name)
                with open(f.name,'rb') as img_file:
                    doc = process_image(img_file)
                    if doc:
                        documents.append(doc)
                        st.session_state.documents.append(doc)
                        st.session_state.uploaded_documents.append(doc)
                        st.sidebar.image(image, caption="Pasted ScreenShot", use_container_width=True)
    except Exception as e:
        st.sidebar.error(f"Error pasting screenshot: {str(e)}")

# Display currently loaded documents
with st.sidebar.expander("Loaded Documents"):
    for doc in st.session_state.uploaded_documents:
        st.write(f"- {doc.metadata.get('source', 'Unnamed document')}")

# Clear documents button
if st.sidebar.button("Clear Documents"):
    st.session_state.documents = []
    st.session_state.uploaded_documents = []
    st.rerun()
# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Assistant:** {message['content']}")
st.markdown("---")

# Clear chat button
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.documents = []
        st.rerun()

# Create the chat interface

user_query = st.text_input("Ask your question:", key="user_input",on_change=clear_input if st.session_state.submit_clicked else None)
submit_button = st.button("Submit", on_click=handle_submit)
if st.session_state.submit_clicked:
    try:
        # Show user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user", 
            "content": user_query
        })
        
        # Show "thinking" message
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.text("Thinking...")
        
        # Format chat history for the workflow
        formatted_history = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in st.session_state.chat_history[-5:-1]  # Exclude current query
        ])
        
        # Run the workflow with chat history
        try:
            response = get_response(QueryModel(query=user_query),HisModel(history=formatted_history),documents = st.session_state.documents if st.session_state.documents else None)
            # Update thinking message with actual response
            thinking_placeholder.write(response[-1])
        except Exception as e :
            print("There is an issue at invoking the response ",f"error : {e}")

        # st.experimental_rerun()
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": response[-1]
        })
        clear_input()
        # st.session_state.submit_clicked = False
        # st.session_state.user_query = ""
        # Debug information
        with st.sidebar:
            st.write("Debug Info:")
            st.write("Chat History Length:", len(st.session_state.chat_history))
            st.write("Last Response:", response[:-1])
            st.write("Number of processed documents:", len(documents))

            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        # Remove the failed interaction from history
        if st.session_state.chat_history:
            st.session_state.chat_history.pop()
            
# Cleanup temporary files when the session ends
def cleanup_temp_files():
    for doc in documents:
        if 'url' in doc.metadata and os.path.exists(doc.metadata['url']):
            os.unlink(doc.metadata['url'])

# Register the cleanup function to run when the Streamlit script reruns
st.session_state['cleanup'] = cleanup_temp_files

