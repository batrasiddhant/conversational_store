from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
# import docx
import re
from agent import *
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),           # logs to console
        logging.FileHandler("myapp.log")   # logs to file named myapp.log
    ]
)
logger = logging.getLogger('app')
logger.info("This should appear in both console and log file")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None

class ChatQuery(BaseModel):
    query: str
    conversation_history: Optional[List[Dict[str, str]]] = None

@app.post("/api/search")
async def search(query: Query):
    try:
        print("--------------------------search called!!---------------------------")
        output =  agent_main(query)
        global recommended_products, clarification_questions, follow_up_question, informational_answer, input_type, answer
        recommended_products = output['recommended_products']
        clarification_questions = output['clarification_questions']
        follow_up_question = output['follow_up_question']
        informational_answer = output['informational_answer']
        input_type = output['input_type']
        answer = """"""
        match input_type:
            case 'keyword':
                for i in recommended_products:
                    answer = answer + i['name'] + " : " + i['justification'] + "\n"
                answer = answer + follow_up_question
            case 'vague':
                answer = "\n".join(clarification_questions)
            case 'question':
                answer = informational_answer
            case _:
                answer = ""
        
        results = []
        keys = ['product_id','name','category','description','top_ingredients','tags','price','margin' ,'justification']
        for i in recommended_products:
            results.append({k : i[k] for k in keys})
        recommended_products = results
        return {
            "results": results,
            "summary": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(query: ChatQuery):
    try:
        global recommended_products, clarification_questions, follow_up_question, informational_answer, input_type, answer
        user_query = query.query
        output =  agent_main(user_query)
        # print("1")
        recommended_products = output['recommended_products']
        # print("2")
        clarification_questions = output['clarification_questions']
        # print("3")
        follow_up_question = output['follow_up_question']
        # print("4")
        informational_answer = output['informational_answer']
        # print("5")
        input_type = output['input_type']
        # print("6")
        answer = """"""
        match input_type:
            case 'keyword':
                for i in recommended_products:
                    answer = answer + i['name'] + " : " + i['justification'] + "\n"
                answer = answer + follow_up_question
            case 'vague':
                answer = "\n".join(clarification_questions)
            case 'informational':
                answer = informational_answer
            case 'good':
                for i in recommended_products:
                    answer = answer + i['name'] + " : " + i['justification'] + "\n"
            case _:
                answer = ""
        
        results = []
        # print("7")
        keys = ['product_id','name','category','description','top_ingredients','tags','price','margin' ,'justification']
        # print(recommended_products)
        try:
            for i in recommended_products:
                print(i.keys)
                results.append({k : i[k] for k in keys})
            recommended_products = results
        except:
            pass
        print("all ok!!")
        # user_query = query.query
        # global recommended_products, clarification_questions, follow_up_question, informational_answer, input_type, answer
        return {
            "response": answer,
            "needsFollowUp": False,
            "products": recommended_products
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/products")
async def get_products(category: Optional[str] = None):
    try:
        global recommended_products
        
        return recommended_products
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT")), log_level="debug")
