import json
import re
from fastapi import APIRouter, Depends, FastAPI, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.db_models.models import User
from app.utills.auth_utils import get_current_user




MODEL_NAME = "llama3.2-vision:11b-instruct-q4_K_M"
# MODEL_NAME = "llama3.2:1b-instruct-q4_K_M"

FAISS_INDEX_PATH = "app/views/rag/faiss_index"
DATA_FILE_PATH = "app/views/rag/modified_data.json"

router = APIRouter()

embeddings = OllamaEmbeddings(model=MODEL_NAME)

if os.path.exists(FAISS_INDEX_PATH):
    print("Loading existing FAISS index...")
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("FAISS index not found. Creating new index from data...")
    if not os.path.exists(DATA_FILE_PATH):
        raise FileNotFoundError(f"{DATA_FILE_PATH} not found. Please provide the data file.")

    jq_schema = """
.[] | 
{
    content: .expanded_description,
    metadata: {
        id: .ID,
        product_name: .\"Product Name\",
        current_price: .\"Current Price\",
        original_price: .\"Original Price\",
        seller_name: .\"Seller Name\",
        image_links: .image_links,
        seller_link: .\"Seller Link\"
    }
}
"""

    loader = JSONLoader(file_path=DATA_FILE_PATH, jq_schema=jq_schema, text_content=False)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_documents = text_splitter.split_documents(documents)

    texts = [doc.page_content for doc in split_documents]
    metadatas = [{"id": str(uuid.uuid4()), **doc.metadata} for doc in split_documents]

    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("FAISS index created and saved.")

llm = OllamaLLM(model=MODEL_NAME)

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10  

@router.post("/query-vectordb")
async def query_documents(request: QueryRequest,
                          current_user: User = Depends(get_current_user),
                          ):
    """
    Query documents from the FAISS vector store and use the LLM to generate responses.
    """
    try:
        retrieved_docs = vectorstore.similarity_search(request.query, k=request.top_k)

        if not retrieved_docs:
            return {"message": "No relevant documents found."}

        retrieved_content = "\n\n".join(
            [f"- {doc.page_content}" for doc in retrieved_docs]
        )

        id_pattern = re.compile(r'ID: (\d+)')
        unique_ids = id_pattern.findall(retrieved_content)

        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as file:
            data = json.load(file)

        matched_data = [item for item in data if str(item['ID']) in unique_ids]

        return {"matched_data": matched_data}

    except Exception as e:
        return {"error": str(e)}
    

@router.post("/query-new")
async def query_documents(request: QueryRequest,
                          current_user: User = Depends(get_current_user),
                          ):
    """
    Query documents from the FAISS vector store and use the LLM to generate responses.
    """
    try:
        casual_queries = ["hi", "hello", "what's up", "hey", "how are you"]
        if any(greet in request.query.lower() for greet in casual_queries):
            return {"response": "Hello! How can I assist you today?"}

        retrieved_docs = vectorstore.similarity_search(request.query, k=request.top_k)

        if not retrieved_docs:
            return {"message": "No relevant documents found."}

        retrieved_content = "\n\n".join(
            [f"- {doc.page_content}" for doc in retrieved_docs]
        )

        prompt = (
            f"""
            You are E-comm Search Engine created by TKGL, an AI assistant for giving responses as a JSON array of IDs only.

            Given the following retrieved content and user query, return a list of IDs that best match the query. The list should be in the format of a JSON array of IDs.

            Retrieved Content:
            {retrieved_content}

            User Query: {request.query}

            Example response format:
            [1, 2, 3]

            If there are no matching products, return an empty list.
            """
        )

        response = llm.invoke(prompt)
        print('LLM Response:')
        print(response)

        json_array_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_array_match:
            json_array_str = json_array_match.group(0)
        else:
            return {"error": "LLM response does not contain a valid JSON array."}

        if not json_array_str.strip():
            return {"error": "LLM response is empty or invalid."}

        try:
            matched_ids = json.loads(json_array_str)
        except json.JSONDecodeError as e:
            return {"error": f"Failed to decode JSON: {str(e)}"}

        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as file:
            data = json.load(file)

        matched_data = [item for item in data if item['ID'] in matched_ids]

        print('Matched Data:')
        print(matched_data)

        return {"matched_data": matched_data, "response": json_array_str}

    except Exception as e:
        return {"error": str(e)}

@router.post("/query")
async def query_documents(request: QueryRequest,
                          current_user: User = Depends(get_current_user),
                          ):
    """
    Query documents from the FAISS vector store and use the LLM to generate responses.
    """
    try:
        retrieved_docs = vectorstore.similarity_search(request.query, k=request.top_k)

        if not retrieved_docs:
            return {"message": "No relevant documents found."}

        retrieved_content = "\n\n".join(
            [f"- {doc.page_content}" for doc in retrieved_docs]
        )

        prompt = (
            f"""
            You are E-comm Search Engine created by TKGL, an AI assistant for give response of list of ID  as a json format only like [1, 2, 3].

            Given the following retrieved content and user query, return a list of IDs that best match the query. The list should be in the format of a JSON array of IDs.

            Retrieved Content:
            {retrieved_content}

            User Query: {request.query}

            Example response format:
            [1, 2, 3]

            If there are no matching products, return an empty list.
            """
        )

        response = llm.invoke(prompt)
        print('LLM Response:')
        print(response)

        json_array_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_array_match:
            json_array_str = json_array_match.group(0)
        else:
            return {"error": "LLM response does not contain a valid JSON array."}

        if not json_array_str.strip():
            return {"error": "LLM response is empty or invalid."}

        try:
            matched_ids = json.loads(json_array_str)
        except json.JSONDecodeError as e:
            return {"error": f"Failed to decode JSON: {str(e)}"}

        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as file:
            data = json.load(file)

        matched_data = [item for item in data if item['ID'] in matched_ids]

        print('Matched Data:')
        print(matched_data)

        return {"matched_data": matched_data,"response": json_array_str}

    except Exception as e:
        return {"error": str(e)}





@router.websocket("/ws/query/")
async def websocket_query(websocket: WebSocket
                            #current_user: User = Depends(get_current_user),
                          ):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request = QueryRequest.parse_raw(data)
            print(request)

            retrieved_docs = vectorstore.similarity_search(request.query, k=request.top_k)

            if not retrieved_docs:
                await websocket.send_text("No relevant documents found.")
                continue

            retrieved_content = "\n\n".join(
                [f"- {doc.page_content}" for doc in retrieved_docs]
            )
            prompt = (
                f"Based on the following Product information: \n\n{retrieved_content}\n\n"
                f"Answer this question's best match:\n\n{request.query}"
            )

            response = llm.invoke(prompt)
            await websocket.send_text(response)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_text(f"Error: {str(e)}")

@router.get("/")
async def root():
    return {"message": "FAISS and LLM-based query service is running."}
