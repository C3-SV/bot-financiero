import os
from openai import OpenAI as OpenAI

from fastapi import FastAPI
from pydantic import BaseModel
import requests

from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI as OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate

# Importar save_feedback
from models.llm_config import save_feedback, prepare_fine_tunning_data
from tools.abaco_platform.abaco_client_tool import set_empresa_id, cargar_datos_empresa_global
from is_client import set_is_abaco_client, get_is_abaco_client

#prepare_fine_tunning_data()
import financebot
import uvicorn
agente = financebot.get_agent()
get_agent_temperature = financebot.get_agent_temperature
set_agent_temperature = financebot.set_agent_temperature

interaction_history = []

app = FastAPI()

port = int(os.environ.get("PORT", 8080))
uvicorn.run(app, host="0.0.0.0", port=port)

temperatura_agente = get_agent_temperature()
band_regenerar = False
status_cliente = False

def regenerar_true():
    global band_regenerar
    band_regenerar = True

class RequestData(BaseModel):
    status_cliente: bool
    empresa_id: str
    text: str
class FeedbackData(BaseModel):
    interaction_id: int
    feedback: str
class ResponseData(BaseModel):
    response: str

@app.post("/process", response_model=ResponseData)
async def process_text(data: RequestData):
    global band_regenerar, temperatura_agente
    global status_cliente, empresa_id

    status_cliente = data.status_cliente
    if (status_cliente != get_is_abaco_client()):
        set_is_abaco_client(status_cliente)
        
    agente = financebot.get_agent()

    if status_cliente:
        empresa_id = data.empresa_id
        set_empresa_id(empresa_id)
        cargar_datos_empresa_global()

    es_cliente = get_is_abaco_client()

    response = agente.run(data.text)
    interaction_history.append({"input": data.text, "response": response})
    
    return ResponseData(response=response)

@app.post("/feedback")
async def provide_feedback(data: FeedbackData):
    global temperatura_agente, band_regenerar
    global status_cliente, empresa_id
    
    if data.interaction_id >= len(interaction_history):
        return {"error": "Invalid interaction_id"}
    
    last_interaction = interaction_history[data.interaction_id]
    user_input = last_interaction["input"]
    last_response = last_interaction["response"]

    if data.feedback.lower() == "regenerar":
        new_temp = temperatura_agente + 0.3
        set_agent_temperature(new_temp)
        regenerar_true()

        new_response =  agente.run(user_input)

        set_agent_temperature(temperatura_agente)
        band_regenerar = False

        interaction_history[data.interaction_id]["response"] = new_response
        return {"interaction_id": data.interaction_id, "new_response": new_response}

    # like y dislike
    if data.feedback.lower() in ["like", "dislike"]:
        save_feedback(user_input, last_response, data.feedback)
        return {"message": f"Retroalimentación '{data.feedback}' guardada."}
    
    return {"error": "Invalid feedback option"}