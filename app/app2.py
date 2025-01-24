import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.tools import tool
from operator import itemgetter
from dotenv import load_dotenv
from typing import List,Dict,Annotated 
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from langchain.llms import HuggingFaceEndpoint
import functools
import os
import pypdf
from PIL import Image 
import pytesseract
from youtubesearchpython import VideosSearch
import logging
import traceback
import torch.nn.functional as F
load_dotenv()
# groq_api_key = os.environ['GROQ_API_KEY']
# google_api_key = os.environ['GOOGLE_API_KEY']
# google_cse_id = os.environ['GOOGLE_CSE_ID']

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

google_api_key = os.getenv('GOOGLE_API_KEY')
groq_api_key = "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"#os.getenv('GROQ_API_KEY')
google_cse_id = os.getenv('GOOGLE_CSE_ID')
endpoint_url = "http://localhost:8080/generate"
llama = ChatGroq(groq_api_key = groq_api_key,
               model_name = "llama3-8b-8192",
               max_tokens = 1200)
print("ahahahahahahahah\n\n\n\n\n ",groq_api_key)

# llama = HuggingFaceEndpoint(
#     endpoint_url=endpoint_url,
#     max_new_tokens=128,  # Maximum number of tokens to generate
#     temperature=0.7,     # Sampling temperature
# )
from langchain.utilities.google_search import GoogleSearchAPIWrapper

web_search_tool = GoogleSearchAPIWrapper(google_api_key=google_api_key,google_cse_id=google_cse_id)
total_tokens_used = 0


class AgentState(TypedDict):
    messages : Annotated[list,add_messages]
    documents: list
    decision: str
    steps: list
    iteration_count: int
    repeat_cot: bool
    chat_history: str
AgentGraph = StateGraph(AgentState)

# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
class HumanMessageWithQueryAndImage(HumanMessage):
    def __init__(self, query: str, image: dict, additional_kwargs=None):
        super().__init__(content=None, additional_kwargs=additional_kwargs)
        self.content = {
            "query": query,
            "image": image
        }

def vector_embedding():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # data_dir = os.path.join(os.path.dirname(current_dir), "data")
    data_dir = os.path.join(current_dir, "data")
    
    # print(f"Current directory: {current_dir}")
    # print(f"Data directory: {data_dir}")
    # print(f"Files in data directory: {os.listdir(data_dir)}")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    loader = PyPDFDirectoryLoader(data_dir) # Data Ingestion
    docs  = loader.load() # Document loading
    if not docs:
        raise ValueError("No documents found in ./data directory")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap=100) # CHunk creation
    final_documents = text_splitter.split_documents(docs)
    if not final_documents:
        raise ValueError("No documents after splitting")
    # chroma_db = Chroma.from_documents(documents = final_documents,
    #                                   collection_name = 'ODP_db',
    #                                   embedding = embeddings,
    #                                   persist_directory = "Vectordb") 
    
    faiss_db = FAISS.from_documents(documents = final_documents,
                                    embedding = embeddings)
    return faiss_db
vector_store = vector_embedding()


def ensure_document(input_data):
    # If already a Document, return as-is
    if isinstance(input_data, Document):
        print("---- IT IS HOW WE WANT IT ----")
        return input_data
    
    # If input is a string, wrap it in a Document
    elif isinstance(input_data, str):
        print("---- IT IS A STRING ----")
        return Document(page_content=input_data, metadata={})
    
    # If input is a dictionary with 'page_content', convert it
    elif isinstance(input_data, dict) and 'page_content' in input_data:
        print("---- IT IS A DICTIONARY ----")
        return Document(
            page_content=input_data.get('page_content', ''),
            metadata=input_data.get('metadata', {})
        )
    elif isinstance(input_data, list):
        print("---- IT IS A LIST ----")
        return [Document(page_content=doc, metadata={}) for doc in input_data]
    
    # Raise an error for unsupported types
    else:
        raise ValueError("Input data must be of type str, dict with 'page_content', or Document.")

def doc_rag(documents):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800,chunk_overlap=100)
    # docs = ensure_document(documents)
    # final_documents = text_splitter.split_documents(docs)
    # print("HERE ARE THE DOCS",documents[:4])
    faiss_db = FAISS.from_documents(documents = documents,
                                    embedding = embeddings)
    print("---- DOC RAG CREATED BABY ----")
    return faiss_db

# import pyimgur

# def upload_to_imgur(image_path, client_id):
#     im = pyimgur.Imgur(client_id)
#     uploaded_image = im.upload_image(image_path, title="Uploaded via Python")
#     return uploaded_image.link


@tool("image_analysis")
def image_analysis(query, image):
    """
    Given an image path and a query, return a description of the image correlated with the query
    """
    llamaQ = ChatGroq(groq_api_key = groq_api_key,
               model_name = "llama3-8b-8192",
               temperature=0.5,
               max_tokens=100)
    llmV = ChatGroq(
                model="llama-3.2-90b-vision-preview",
                temperature=0.7,
                max_tokens=300)
    # llamaQ = HuggingFaceEndpoint(
    # endpoint_url=endpoint_url,
    # max_new_tokens=128,  # Maximum number of tokens to generate
    # temperature=0.7,     # Sampling temperature
    # )
    # llmV = HuggingFaceEndpoint(
    # endpoint_url=endpoint_url,
    # max_new_tokens=300,  # Maximum number of tokens to generate
    # temperature=0.7,     # Sampling temperature
    # )

    prompt = """Provide a detailed point wise query from the query giving by the user that can be helpful to get rich image descriptions HERE IS THE QUERY : {query}"""
    query_rephraser  = ({itemgetter("query") : "query"}|prompt|llamaQ|StrOutputParser())
    new_query = query_rephraser.run({"query":query})
    system_prompt = """You are an image analysis AI tasked with providing a detailed description of the image based on the user's query.
    """
    user_message = HumanMessageWithQueryAndImage(
    query=new_query,
    image={
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{image}"
        }
    }
    )
    chat_prompt = ChatPromptTemplate.from_messages([
        system_prompt,  # System prompt
        MessagesPlaceholder(variable_name="messages")  # Placeholder for user and other messages
    ])
    formatted_prompt = chat_prompt.format(
        messages = [
                    {"role": "user", "content": user_message.content}
        ]
    )
    # vis_prompt = ChatPromptTemplate.from_template(sys_message)
    # vis_llm = (formatted_prompt| llmV | StrOutputParser())
    response = llmV(formatted_prompt)|StrOutputParser()
    print("--- image analysis done ---")
    return response
    # if image:
    #     result = vis_llm.run({"image": image, "query": new_query})
    #     return result

    # {
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/jpeg;base64,{base64_image}"
    #                     }
    #                 }
    #             ]
    #         }
    #     ]
    # }
    # vis_prompt = ChatPromptTemplate.from_message(sys_message)
    logger.error("No image URL found in the documents.")

    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", sys_message),
    #     ("human", f"Query: {new_query}")
    # ])
    # llm = ChatGroq(
    # model="llama-3.2-11b-vision-preview",
    # temperature=0.5,
    # )
    # chain = prompt | llm | StrOutputParser()
    # outputs = chain.invoke({"query": new_query})
    # return outputs



@tool("youtube_search")
def youtube_search(query:str, max_results: int = 4):
    """
    Searches for videos on YouTube based on a query
    Args:
        query: The search query string
        max_results: Maximum number of results to return (default is 4)
    Returns:
        str: List of video titles and URLs
    """
    try:
        results = VideosSearch(query, limit=max_results).result()['result']
        formatted_results = []
        for video in results:
            title = video['title']
            url  = video['link']
            formatted_results.append(f"{title} - {url}")
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error searching YouTube: {str(e)}"

@tool("rag_search")
def rag_search(query: str):
    """
    Searches through the document database using RAG (Retrieval Augmented Generation).
    
    Args:
        query: The search query string
    
    Returns:
        str: Relevant document content from the vector store
    """
    retrieved_docs = vector_store.similarity_search(query,k=4)
    # retrieved_docs = Document(page_content=retrieved_docs)
    return retrieved_docs

@tool("web_search")
def web_search(query: str):
    """
    Performs a web search using Google Search API.
    
    Args:
        query: The search query string
    
    Returns:
        str: Search results from the web
    """
    print("--web search--")
    web_results = web_search_tool.run(query)
    web_results = Document(page_content=web_results)
    print("-- no error --")
    return web_results

def create_context_chuncks(state: AgentState):
    """
    Given a hoad of documents, split them into smaller chunks for better processing
    """
    if not state["documents"]:
        return state
    documents = state["documents"]
    doc_llm = ChatGroq(groq_api_key = groq_api_key,
               model_name="llama3-8b-8192",
               max_tokens=500)
    # doc_llm = HuggingFaceEndpoint(
    # endpoint_url=endpoint_url,
    # max_new_tokens=500,  # Maximum number of tokens to generate
    # temperature=0.7,     # Sampling temperature
    # )
    prompt1 = """
                You are an expert at identifying the necessary key points to all the documents provided 
                 - You must keep track of the context that the document suggest and preserve all heading's suchs as titles , names and all other specification as it is and summarize the rest
                 - You cannot miss out on any important information that the document provides
                 - You should also make your summary as concise as possible but only if you think the document is small 
                 - If it figures that the document is large try to retail as much information as possible
                 
                 Here is the context:  {context}"""
    prompt1 = ChatPromptTemplate.from_template(prompt1)
    sum_doc_chain = ({"context": itemgetter("documents")} | prompt1 | doc_llm | StrOutputParser())
    documents = sum_doc_chain.invoke({"documents":documents})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.create_documents([documents])
    documents = text_splitter.split_documents(texts)
    vector_db = doc_rag(documents)
    rag_docs = vector_db.similarity_search(state["messages"][-1].content,k=4)
    state["documents"] = format_docs(rag_docs)
    print("--- context chuncks created ---")
    return state


def judge_response(state: AgentState):
    # faiss_db = doc_rag(state["documents"])
    # context = faiss_db.similarity_search(state["messages"][-1].content,k=4)
    prompt = """
        You are a judge evaluating the quality of responses generated by an AI assistant with regards to the context provided. 
        Your task is to assess whether the RESPONSE sufficiently addresses the QUESTION based on these criteria:
        1. Completeness: Does it fully answer all parts of the question?
        2. Relevance: Is it directly related to the question?
        3. Clarity: Is it clear and logically structured?
        4. Accuracy: Is it factually correct?

        QUESTION:
        {question}

        RESPONSE:
        {response}

        CONTEXT:
        {context}

        Based on your evaluation, provide one of these labels:
        - "Sufficient": If the response meets all criteria.
        - "Insufficient": If it fails any of the criteria.
        Make your response as concise and short as possible
        """
    question = state["messages"][-1].content
    previous_response = state["steps"][-1] if state["steps"] else "No prior steps available."

    prompt = ChatPromptTemplate.from_template(prompt)
    judge_llm = (
        {
            "question" : itemgetter("question"),
            "response": itemgetter("response"),
            "context": itemgetter("context")
        }
        |
        prompt
        |
        llama
        |
        StrOutputParser()
    )
    judgment = judge_llm.invoke({"question":question,"response":previous_response,"context":state["documents"]})
    print(f"--- judge response {judgment}---")
    if "Sufficient" in judgment:
        # print("final answer : ",state["steps"][-1])
        state["repeat_cot"] = False
        # state["steps"].append("Judge LLM: Response is sufficient. Proceeding.")
    elif "Insufficient" in judgment:
        state["repeat_cot"] = True
        state["steps"].append(f"Judge LLM: Response is insufficient. Reason - {judgment}")
    # else:
    #     state["repeat_cot"] = False
    #     state["steps"].append("Judge LLM: Response is insufficient. Reason - {judgment}")
    
    return state#{"repeat_cot": state["repeat_cot"], "steps": state["steps"]}

def chain_of_thought_decision(state: AgentState):

    query = state['messages'][-1].content
    documents = state['documents']

    llm = ChatGroq(groq_api_key = groq_api_key,
               model_name="llama3-8b-8192",
               max_tokens=500)
    # llm = HuggingFaceEndpoint(
    # endpoint_url=endpoint_url,
    # max_new_tokens=500,  # Maximum number of tokens to generate
    # temperature=0.7,     # Sampling temperature
    # )

    prompt = """You are a precise decision-making agent that must output EXACTLY ONE of these words: "web_search", "rag_search", "both", "general", "image_analysis", "youtube_search", "all_three", "all_four".
    
    Analyze the query and decide based on these strict criteria:

    Query : {question}
    Context: {context}

    1. Output "web_search" if the query:
        • Contains phrases like "deeply explain", "elaborate", "comprehensive", "in detail", "explain thoroughly"
        • Requires current/real-time information (e.g., "latest", "recent", "current", "now", "today")
        • Asks about news, trends, or ongoing events
        • Asks to search the internet or search the web 
        • Needs information beyond a company's internal knowledge
        • Requires comparative analysis with other companies or technologies

    2. Output "rag_search" if the query:
        • Specifically mentions Oman Data Park's products/services:
            - ODP
            - odp
            - Nebula AI
            - Cloud services
            - Data center
            - Hosting solutions
            - Cybersecurity services
            - Oman Data Park
        • Asks about company-specific information:
            - ODP pricing
            - Service offerings
            - Infrastructure details
            - Company policies
            - Business partnerships

    3. Output "both" if the query:
        • Combines ODP-specific information with broader industry context
        • Requests comparison between ODP services and market alternatives
        • Contains both internal (ODP) and external reference points
        • Needs both company data and market research
        • Requires understanding of ODP's position in the larger market

    4. Output "general" if the query:
        • Is a basic greeting ("hi", "hello", "how are you")
        • Asks for simple definitions without specific context
        • Can be answered with common knowledge
        • Is a clarifying question about the conversation
        • Doesn't require specific company knowledge or current information
        • If there is a document provided by the user he will ask question like "summarize the document","resume","given context"
        • If the user asks questions like "what does this say really","what is this about","what is this documents about"

    5. Output "image_analysis" if the query:
        • References an uploaded image
        • Asks about image content
        • Requests visual analysis
        • Contains phrases like "in this image", "from this picture", "from the given image"

    6. Output "youtube_search" if the query:
        • Contains phrases like "youtube", "video", "youtube video", "youtube search"
        • Asks for a video or a list of videos
        • Requests information from a video or a list of videos

    7. Output "all_three" if the query:
        • Contains phrases like "from the given image" and "give an extensive explanation" and if it's about ODP
        • Asks for a list of images, web search, and RAG 
        • Suggests using an extensive search along with image analysis if any images are provided 
        
    8. Output "all_four" if the query:
        • Contains phrases like "provide me a youtube link" while there also being a reference to ODP and if the user asks for extensive explanations on top of the image provided
        • Asks for a list of videos, images, web search, and RAG

    9. Output "web/video" if the query:
        • Indicates the need for both extensive web search for more information and specific YouTube links
        • Contains phrases like "detailed web search and YouTube links", "comprehensive search and video links", "web and video search"

    Remember: You must output ONLY ONE of these exact words: "web_search", "rag_search", "both", "general", "image_analysis", "youtube_search", "all_three", "all_four", "web/video".
    No other words or explanations are allowed in your response.
    """
    decision_prompt = ChatPromptTemplate.from_template(prompt)
    llm_decision = ({"question" : itemgetter("query"),"context" : itemgetter("documents")}|decision_prompt|llm|StrOutputParser())

    val = llm_decision.invoke({"query":query,"documents":documents})
    if "web" in val:
        decision = "web_search"
        reasoning_step = "Query requires real-time information; invoking web search."
    elif "rag" in val:
        decision = "rag"
        reasoning_step = "Query requires detailed information; invoking RAG."
    elif "both" in val:
        decision = "both"
        reasoning_step = "Query requires both real-time and knowledge base information; invoking both tools."
    elif "image" in val:
        decision = "image_analysis"
        reasoning_step = "Query requires image analysis; invoking image analysis tool."
    elif "youtube" in val:
        decision = "youtube_search"
        reasoning_step = "Query requires YouTube search; invoking YouTube search tool."
    elif "all three" in val:
        decision = "all_three"
        reasoning_step = "Query requires all three tools; invoking image,web and RAG"
    elif "all four" in val:
        decision = "all_four"
        reasoning_step = "Query requires all four tools; invoking image,web,RAG and youtube"
    elif "web and video" in val:
        decision = "web/video"
        reasoning_step = "Query requires all four tools; invoking web and youtube"

    else:
        decision = "general"
        reasoning_step = "Query can be answered generally without external tools."

    state["decision"] = decision
    state["steps"].append(f"Tool LLM : {reasoning_step}")
    print(f"--- Decision : {decision}    {val} ---")
    return state#{"decision": state["decision"], "steps": state["steps"]}

def format_docs(docs):
    try:
        # If it's a single Document object
        if isinstance(docs, Document):
            return docs.page_content
            
        # If it's a list of Document objects
        if isinstance(docs, list):
            try:
                val = "\n".join(doc["page_content"] for doc in docs)
            except:
                val = "\n".join(doc.page_content for doc in docs)
            return val
            
        # If it's a string
        if isinstance(docs, str):
            return docs
            
        print(f"Unexpected type: {type(docs)}")
        return str(docs)
            
    except Exception as e:
        print(f"Error in format_docs: {type(docs)} {docs[:3]}")
        raise

def invoke_tools(state: AgentState):
    if state["decision"] == "rag":
        documents = rag_search(state["messages"][-1].content)
        state["documents"] = format_docs(documents)
        state["steps"].append("Retrieved documents using RAG.")
    
    elif state["decision"] == "web_search":
        documents = web_search(state["messages"][-1].content)
        state["documents"] = format_docs(documents)
        state["steps"].append("Retrieved documents using web search.")
    
    elif state["decision"] == "both":
        rag_docs = rag_search(state["messages"][-1].content)
        web_docs = web_search(state["messages"][-1].content)
        state["documents"] = format_docs(rag_docs) + format_docs(web_docs)
        state["steps"].append("Retrieved documents using both RAG and web search.")
    
    elif state["decision"] == "image_analysis":
        image_base64 = None
        for doc in state["documents"]:
            metadata = doc.get("metadata", {}) if isinstance(doc, dict) else getattr(doc, "metadata", {})
            if metadata.get("type") == "image":
                image_base64 = metadata.get('image_file')
                logger.debug(f"Processing image URL: {image_base64}")
                break
        if not image_base64 :
            state["steps"].append("No valid image found for analysis.")
        else:
            try:
                state["documents"] = format_docs(image_analysis(state["messages"][-1].content,image_base64))
                # state["steps"].append(f"Successfully analyzed image at path: {image_url}")
            except Exception as e:
                state["steps"].append(f"Error analyzing image: {str(e)}")
    elif state["decision"] == "youtube_search":
        youtube_docs = youtube_search(state["messages"][-1].content)
        state["documents"] = format_docs(youtube_docs)
        state["steps"].append("Retrieved documents using youtube search.")
    
    elif state["decision"] == "all_three":
        image_docs = image_analysis(state["messages"][-1].content,image_base64)
        rag_docs = rag_search(state["messages"][-1].content)
        web_docs = web_search(state["messages"][-1].content)
        state["documents"] = format_docs(image_docs) + format_docs(rag_docs) + format_docs(web_docs)
        state["steps"].append("Retrieved documents using image analysis,RAG and web search.")
    
    elif state["decision"] == "all_four":
        image_docs = image_analysis(state["messages"][-1].content,image_base64)
        rag_docs = rag_search(state["messages"][-1].content)
        web_docs = web_search(state["messages"][-1].content)
        youtube_docs = youtube_search(state["messages"][-1].content)
        state["documents"] = format_docs(image_docs) + format_docs(rag_docs) + format_docs(web_docs) + format_docs(youtube_docs)
        state["steps"].append("Retrieved documents using image analysis,RAG,web search and youtube search.")
    elif state["decision"] == "web/video":
        web_docs = web_search(state["messages"][-1].content)
        youtube_docs = youtube_search(state["messages"][-1].content)
        state["documents"] = format_docs(web_docs) + format_docs(youtube_docs)
        state["steps"].append("Retrieved documents using web search and youtube search.")
    else:
        # No external tools invoked for general answers
        if state["documents"]:
            state["documents"] = format_docs(state["documents"])
        state["steps"].append("No external tools invoked; providing a general answer.")
    
    return state#{"documents": state["documents"], "steps": state["steps"]}


def generate_answer(state: AgentState):
    # context = state["documents"]
    question = state["messages"][-1].content if state["messages"] else ""
    llama_ans = ChatGroq(groq_api_key = groq_api_key,
                model_name = "llama3-8b-8192",
                max_tokens = 3000)
    # llama_ans = HuggingFaceEndpoint(
    # endpoint_url=endpoint_url,
    # max_new_tokens=500,  # Maximum number of tokens to generate
    # temperature=0.7,     # Sampling temperature
    # )

    doc_llm = ChatGroq(groq_api_key = groq_api_key,
               model_name="llama3-8b-8192",
               max_tokens=1000)
    # doc_llm = HuggingFaceEndpoint(
    # endpoint_url=endpoint_url,
    # max_new_tokens=1000,  # Maximum number of tokens to generate
    # temperature=0.7,     # Sampling temperature
    # )
    prompt1 = """
                You are an expert at identifying the necessary key points to all the documents provided 
                 - You must keep track of the context that the document suggest and preserve all heading's suchs as titles , names and all other specification as it is and summarize the rest
                 - You cannot miss out on any important information that the document provides
                 - You should also make your summary as concise as possible but only if you think the document is small 
                 - If it figures that the document is large try to retail as much information as possible
                 
                 Here is the context:  {context}
                 Please try to limit the size of the summary to a maximum bound of 700"""
    prompt1 = ChatPromptTemplate.from_template(prompt1)
    sum_doc_chain = ({"context": itemgetter("documents")} | prompt1 | doc_llm | StrOutputParser())
    print("--- Doc Summarized ---")
    context = sum_doc_chain.invoke({"documents":state["documents"]})
    
    SYS_PROMPT = """You are an expert summarizer assessing all provided context if given.
                You should maintain context of the conversation and provide relevant answers.
                Follow these instructions:
                    - Consider the chat history for context
                    - If the document potentially contains multiple headings convert these heading into individual headlines to a brief paragraph
                    - Your summary should have a strong introduction covering the overall idea of the answer and it should also contain a conclusion reiterating the relevant points
                    - Your summary should be in accordance with the documents emphasizing on all available points
                    - Always check if there is any context provided and write a relevant answer on the basis of that conetext itself
                If there is no context then just give a brief response donot mention that context was not provided,provide a very friendly and welcoming answer futher tell the user that they can ask questions about ODP
                 - dont respond with "I don't have any context to work with yet."
                 - dont ask for context

                When providing answers:
                 - If YouTube links are available, integrate them naturally into your response
                 - Format links as markdown: [Title](URL)
                 - If web search results contain useful links, include them as references
                 - For image analysis, describe what you see and relate it to the user's question
                Things to note:
                 - If the user askes about ODP which stands for Oman Data Park so just talk about Oman Data Park
                 
            """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYS_PROMPT),
        ("human", """Chat History: {chat_history} Context: {context} Current Question: {question}""")
    ])

    # chat_history = ""
    # if hasattr(st.session_state, 'chat_history'):
    #     chat_history = "\n".join([f"{msg['role']}: {msg['content']}" 
    #                              for msg in st.session_state.chat_history[:-1]])
    resp_llm = (
        {
            "context": itemgetter("context"),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history")#RunnableLambda(lambda x: chat_history)
        }
        |
        prompt
        |
        llama_ans
        |
        StrOutputParser()
    )
    
    answer = resp_llm.invoke({
        "context": context,
        "question": question,
        "chat_history": state.get("chat_history","")
    })
    
    state["iteration_count"] += 1
    state["steps"].append(answer)
    print("--- generating answer ---")
    print("-- no error--")
    return state


AgentGraph.add_node("chain_of_thought_decision", chain_of_thought_decision)
AgentGraph.add_node("judge_response", judge_response)
AgentGraph.add_node("invoke_tools", invoke_tools)
AgentGraph.add_node("generate_answer", generate_answer)
AgentGraph.add_node("create_context_chuncks", create_context_chuncks)

# AgentGraph.set_entry_point("chain_of_thought_decision")
AgentGraph.set_entry_point("create_context_chuncks")
# Add edges for the main flow
AgentGraph.add_edge("create_context_chuncks","chain_of_thought_decision")
AgentGraph.add_edge("chain_of_thought_decision", "invoke_tools")
AgentGraph.add_edge("invoke_tools", "generate_answer")

# Add conditional edges from judge_response
def repeat_required(state: AgentState):
    if state["repeat_cot"]:
        print("--- repeat required ---")
        return "chain_of_thought_decision"
    else:
        print("--- end ---")
        return "end"
AgentGraph.add_conditional_edges(
        "judge_response",
        repeat_required,
        {"chain_of_thought_decision":"chain_of_thought_decision","end":END}
)

def is_count_reached(state: AgentState):
    print(state["iteration_count"])
    if state["iteration_count"]<3:
        return "judge_response"
    else:
        print("Count reached")
        print("--- end ---")
        return "end"
AgentGraph.add_conditional_edges(
        "generate_answer",
        is_count_reached,
        {"judge_response":"judge_response","end":END}
        )
# workflow = AgentGraph.compile()

class WorkflowManager:
    def __init__(self):
        self.workflow = AgentGraph.compile()
    def process_query(self, query: str,chat_history: str = "", documents: List[Dict] = []):
        try:
            logger.debug(f"Processing query: {query}")
            logger.debug(f"Documents received: {documents}")
            # Store documents if provided


            # If no query but documents exist, wait for query
            # if not query and (documents or image_path):
            #     return "Documents received. Please ask your question."
            # print("---------- DOCUMENTS ----------\n\n",documents,'\n\n')

            # State = {
            #     "messages": [{"role": "user", "content": query}],
            #     "documents": documents,
            #     "decision": "",
            #     "steps": [],
            #     "iteration_count": 0,
            #     "repeat_cot": False,
            #     "chat_history": chat_history,
            # }
            # logger.debug(f"Initial state: {State}")
            # print(f"this is the fkn Stateee :\n\n{State}\n")
            response = self.workflow.invoke({
                "messages": [{"role": "user", "content": query}],
                "documents": documents,
                "decision": "",
                "steps": [],
                "iteration_count": 0,
                "repeat_cot": False,
                "chat_history": chat_history,
            })
            # print(f"\n\nDID WE CROSS THIS FKN SHIT?:\n\n{State}\n")
            logger.debug(f"Workflow response: {response}")
            if response and "steps" in response and response["steps"]:
                return response["steps"]  # Return the last step as the answer
            return "No response generated"
        except Exception as e:
            logger.error(f"WorkflowManager error: {str(e)}")
            logger.error(traceback.format_exc())
            raise

workflow_manager = WorkflowManager()
# workflow_manager.process_query(query = "what are the services provided by ODP")

# from docx import Document as DocxDocument
# from langchain.schema import Document
# import os

# # Function to convert a .docx file to langchain.schema.Document
# def convert_docx_to_langchain_document(file_path: str) -> Document:
#     # Read the .docx file
#     doc = DocxDocument(file_path)
    
#     # Extract all text content from the document
#     content = "\n".join([para.text for para in doc.paragraphs if para.text.strip() != ""])
    
#     # Define metadata (optional, can be anything like file name or path)
#     metadata = {
#         "file_name": os.path.basename(file_path),
#         "file_path": file_path
#     }
    
#     # Create and return a langchain Document
#     return Document(page_content=content, metadata=metadata)

# # Example usage
# file_path = "/Users/ayushbhakat/Desktop/Ayush Bhakat CV (2).docx"
# document = convert_docx_to_langchain_document(file_path)
# workflow_manager.process_query("what rating would you give to this resume?", documents=[document])

# Now you can access the content and metadat



# Initialize session state for chat history if not exists

# st.title("🤖 Intelligent Document Assistant")

# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         if message["role"] == "user":
#             st.markdown(f"**You:** {message['content']}")
#         else:
#             st.markdown(f"**Assistant:** {message['content']}")
# st.markdown("---")

# col1, col2 = st.columns([4, 1])
# with col2:
#     if st.button("Clear Chat"):
#         st.session_state.chat_history = []
#         st.rerun()
# # Create the chat interface
# user_query = st.text_input("Ask your question:", key="user_input")

# if user_query:
#     try:
#         # Show user message
#         with st.chat_message("user"):
#             st.write(user_query)
        
#         # Add user message to chat history
#         st.session_state.chat_history.append({
#             "role": "user", 
#             "content": user_query
#         })
        
#         # Show "thinking" message
#         with st.chat_message("assistant"):
#             thinking_placeholder = st.empty()
#             thinking_placeholder.text("Thinking...")
        
#         # Run the workflow
#         response = workflow.invoke({
#             "messages": [{"role": "user", "content": user_query}],
#             "documents": [],
#             "decision": "",
#             "steps": [],
#             "iteration_count": 0,
#             "repeat_cot": False
#         })
        
#         # Get the answer from the steps
#         if response and "steps" in response and response["steps"]:
#             answer = response["steps"][-1]  # Get the last step
            
#             # Update thinking message with actual response
#             thinking_placeholder.write(answer)
            
#             # Add assistant response to chat history
#             st.session_state.chat_history.append({
#                 "role": "assistant", 
#                 "content": answer
#             })
            
#             # Debug information
#             st.sidebar.write("Debug Info:")
#             st.sidebar.write(f"Iteration Count: {response.get('iteration_count', 0)}")
#             st.sidebar.write(f"Steps: {len(response.get('steps', []))}")
#         else:
#             st.error("No response generated")
            
#     except Exception as e:
#         st.error(f"An error occurred: {str(e)}")
#         # Remove the failed interaction from history
#         if st.session_state.chat_history:
#             st.session_state.chat_history.pop()



