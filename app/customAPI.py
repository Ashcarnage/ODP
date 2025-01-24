from fastapi import FastAPI, HTTPException
from model import QueryModel,HisModel
from app2 import workflow_manager
from typing import List,Dict
import traceback
import logging
import json
from langchain.schema import Document

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


app = FastAPI(title="ODP BOT",
              description="ODP BOT implements RAG,Web Search as well as Chain of Thought",
              version="1.0.0")

@app.post("/Agent_Response")
async def Agent_Response(query: QueryModel,chat_history: HisModel,documents: List[Dict] = None):
    try:
        logger.debug(f"Received query: {query.model_dump()}")
        logger.debug(f"Received history: {chat_history.model_dump()}")
        logger.debug(f"Received documents: ")

        processed_docs = []
        if documents:
            for doc in documents:
                if isinstance(doc, dict):
                    processed_docs.append(doc)
                else:
                    processed_docs.append(doc.dict())
        
        response = workflow_manager.process_query(
            query=query.query,
            chat_history=chat_history.history,
            documents=processed_docs
        )
        
        return {"response": response}
    except Exception as e:
        # Log the full error traceback
        logger.error(f"Error processing request: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "traceback": traceback.format_exc(),
                "query": query.model_dump() if query else None,
                "history": chat_history.model_dump() if chat_history else None,
                "documents": documents,
                "documents_count": len(documents) if documents else 0
            }
        )

