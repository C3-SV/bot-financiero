import os
from openai import OpenAI as OpenAI

from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI as OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate

# Importar save_feedback
from models.llm_config import save_feedback, prepare_fine_tunning_data

#prepare_fine_tunning_data()

from tools.abaco_platform.abaco_client_tool import set_empresa_id, cargar_datos_empresa_global

# IMPLEMENTACION UNICAMENTE PARA TESTEO, REQUIERE MAS DESARROLLO
from is_client import set_is_abaco_client, get_is_abaco_client
#! AQUI SE DEFINE TODO LO RELACIONADO A CLIENTE O NO CLIENTE
status_cliente = True
# ESTA VARIABLE DEFINE EL COMPORTAMIENTO, SI ES CLIENTE O NO
set_is_abaco_client(status_cliente)
# AQUI SE DEFINE EL ID DE LA EMPRESA, SI ES CLIENTE
empresa_id = "2-DistribuidoraComercialSur"
# HAY 3 EMPRESAS DISPONIBLES
# 1-TecnologiaInnovadora 
# 2-DistribuidoraComercialSur
# 3-ManufacturasIndustriales 
#! EN EL 3, FALTA EL CASO DE UN CREDITO EN MORA, AGREGAR CONDICIONES ESPECIALES PARA ESTE CASO
es_cliente = get_is_abaco_client()

if es_cliente == True:
    set_empresa_id(empresa_id)
    cargar_datos_empresa_global()

import financebot
get_agent_temperature = financebot.get_agent_temperature
set_agent_temperature = financebot.set_agent_temperature
agente = financebot.get_agent() 
interaction_history = []

band_regenerar = False
temperatura_agente = get_agent_temperature()
def regenerar_true():
    global band_regenerar
    band_regenerar = True
    
def imprimir_respuesta(respuesta):
    print("BOT: ", respuesta)
    # REGENERAR Y FEEDBACK   
    regenerar_respuesta = input("Regenerar? si o no: ").strip().lower() 
    if regenerar_respuesta == "si":
        regenerar_respuesta = True
    else:
        regenerar_respuesta = False
    
    if regenerar_respuesta:
        global temperatura_agente
        new_temp = temperatura_agente + 0.3
        set_agent_temperature(new_temp)
        regenerar_true()
        return  
   
    # Mostrar opciones
    print("Feedback: 'like', 'dislike'")
    feedback = input("Retroalimentacion: ").strip().lower()
    #! ESTA VARIABLE DEBE CAMBIARSE PARA SER TOMADA DEL MENSAJE DE LA API
    # Si no tomo ninguna opcion 
    if feedback not in ['like', 'dislike']:
        return
    # Like y Dislike
    if feedback in ["like", "dislike"]:
        last_interaction = interaction_history[-1]
        last_input = last_interaction["input"]
        last_response = last_interaction["response"] 
        # Funcion de save_feedback en llm_config
        save_feedback(last_input, last_response, feedback)
        print(f"Retroalimentacion guardada: {feedback}")
    
    
    
def run_chatbot(): 
    print(f"from main, ES CLIENTE ABACO: {es_cliente}")
    print("Bienvenido al Chatbot Financiero de Abaco. ¿En qué puedo ayudarte?\nEscribe 'salir' para terminar.")
    global band_regenerar
    global temperatura_agente
    band_regenerar = False
    while True: 
        if band_regenerar == True:
            user_input = user_input
            response = agente.run(user_input)
            set_agent_temperature(temperatura_agente)
            band_regenerar = False
        else: 
            user_input = input("User: ").strip()
            if user_input.lower() == 'salir':
                break
            response = agente.run(user_input)
        
       
        # Ejecutar agente 
        
        # NUEVA FUNCION
        interaction_history.append({"input": user_input, "response": response})
        imprimir_respuesta(response)
        # Mostrar respuesta             


            
            
if __name__ == "__main__":
    # Logica de cliente
    run_chatbot()
