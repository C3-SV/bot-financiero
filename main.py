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
from fastapi.middleware.cors import CORSMiddleware

# Importar save_feedback
from models.llm_config import save_feedback, prepare_fine_tunning_data
from tools.abaco_platform.abaco_client_tool import set_empresa_id, cargar_datos_empresa_global, get_empresa_id
from is_client import set_is_abaco_client, get_is_abaco_client

#prepare_fine_tunning_data()
import financebot
agente = financebot.get_agent()
get_agent_temperature = financebot.get_agent_temperature
set_agent_temperature = financebot.set_agent_temperature

interaction_history = []

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globales
temperatura_agente = get_agent_temperature()
band_regenerar = False
status_cliente = False
empresa_id_global = get_empresa_id()


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
    
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

@app.post("/process", response_model=ResponseData)
async def process_text(data: RequestData):
    global band_regenerar, temperatura_agente
    global status_cliente, empresa_id_global
    global agente
    global interaction_history

    status_cliente = data.status_cliente
    if (status_cliente != get_is_abaco_client()):
        print("El estado del cliente ha cambiado.")
        set_is_abaco_client(status_cliente)
        agente = financebot.get_agent()

    if (empresa_id_global != data.empresa_id and status_cliente):
        print("El id de la empresa ha cambiado.")
        set_empresa_id(data.empresa_id)
        empresa_id_global = data.empresa_id
        cargar_datos_empresa_global()
        agente = financebot.get_agent()
        
    response = agente.run(data.text)
    interaction_history.append({"input": data.text, "response": response})

    return ResponseData(response=f"{response}")

@app.post("/feedback")
async def provide_feedback(data: FeedbackData):
    global temperatura_agente, band_regenerar
    global status_cliente, empresa_id_global
    
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
