import json
import os, sys
from typing import Dict, Any
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.stdout.reconfigure(encoding='utf-8')

# Preguntas tool 
from tools.general.preguntas_tool import tool_preguntas, obtener_retriever_correcto
# Importar el chatbot y memoria
import models.llm_config as llm_config
chat = llm_config.get_openai_llm()
simple_chat = llm_config.get_simple_openai_llm()
get_chat_completion = llm_config.get_chat_completion
get_client_chat_completion = llm_config.get_client_chat_completion

from langchain.chains import LLMChain
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

# Memory
from memory.context import get_conversation_memory
memory = get_conversation_memory()

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # FinanceBot/
DATA_DIR = BASE_DIR / "data" / "abaco_platform"

# Variables globales
empresa_id_global = None
datos_empresa_global = None

def set_empresa_id(empresa_id: str):
    """Establece el ID de la empresa globalmente."""
    global empresa_id_global
    empresa_id_global = empresa_id.strip()
    
def get_empresa_id():
    global empresa_id_global
    return empresa_id_global

def cargar_datos_empresa_global():
    """Carga los datos de la empresa una sola vez y los almacena globalmente."""
    global datos_empresa_global
    if empresa_id_global is None:
        raise ValueError("No se ha establecido el ID de la empresa.")
    file_path = os.path.join(DATA_DIR, f"{empresa_id_global}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            datos_empresa_global = json.load(file)
    else:
        raise FileNotFoundError(f"No se encontró el archivo para la empresa {empresa_id_global}")


def generar_respuesta_general(pregunta: str) -> str:
    """Genera una respuesta general para cualquier usuario con evaluación de relevancia."""
    pregunta_sencilla = "Desde la Web de Abaco, Dame conceptos y pautas generales que ayuden a resolver esta pregunta: " + pregunta
    retriever_correcto = obtener_retriever_correcto(pregunta_sencilla)
    respuesta_blog = retriever_correcto.invoke({"query": pregunta_sencilla})
    
    # Prompt para evaluar relevancia
    relevancia_prompt = f"¿Es esta información relevante para la sigueinte pregunta. Considera que no debe desviarse de la pregunta. Debe aportar a la respuesta. Debe funcionar como una guia para una respuesta mas detallada. Debe ser una base conceptual. Considera que debe ser un marco contextual que aporte a generar una respuesta mas completa\n Pregunta: '{pregunta}'? \nResponde exclusivamente 'si' o 'no'.\nInformación: {respuesta_blog}"
    print("Informacion del blog: ", respuesta_blog)
    relevancia = chat(relevancia_prompt).content.strip().lower()
    
    # Filtrar según relevancia
    if relevancia == 'si':
        return respuesta_blog
    else:
        return "No hay información específica en la web de Abaco para esta pregunta."

def generar_recomendacion_personalizada(pregunta: str, respuesta_general: str) -> str:
    """Genera una recomendación personalizada para clientes de Ábaco."""
    if datos_empresa_global is None:
        raise ValueError("No se han cargado los datos de la empresa.")
    
    try:
        informacion_empresa = json.dumps(datos_empresa_global, indent=2, ensure_ascii=False)
    except TypeError as e:
        print(f"Error al convertir JSON a string: {e}")
        informacion_empresa = "No se pudo cargar la información de la empresa."
    
    #! AQUI IRA EL NUEVO CHAT COMPLETION PARA CLIENTES 
    respuesta_personalizada = get_client_chat_completion(
        prompt=pregunta,
        general_answer=respuesta_general if respuesta_general != "No hay información específica en la web de Abaco para esta pregunta." else None,        
        chat_history=memory.load_memory_variables({})["chat_history"],
        empresa_data=informacion_empresa
    )
    return respuesta_personalizada
    
    

def abaco_client(pregunta: str) -> str:
    """Herramienta para clientes de Ábaco que usa variables globales."""
    if empresa_id_global is None:
        raise ValueError("No se ha establecido el ID de la empresa.")
    respuesta_general = generar_respuesta_general(pregunta)
    respuesta_final = generar_recomendacion_personalizada(pregunta, respuesta_general)
    return respuesta_final

tool_abaco_client = Tool(
    name="Abaco_client",
    func=abaco_client,
    description="Responder preguntas para clientes de Abaco"
)
print("Tool Abaco Client cargado")

# Ejemplo de uso
if __name__ == "__main__":
    # Configurar la sesión del usuario
    empresa_id = "1-TecnologiaInnovadora"
    set_empresa_id(empresa_id)
    cargar_datos_empresa_global()
    
    # Preguntas de ejemplo
    #pregunta_usuario = "¿Cómo puedo mejorar mi liquidez?"
    #print(abaco_client(pregunta_usuario))
    
    pregunta_usuario_creditos = "¿Que puedes decirme de mis ultimos 3 meses en mi empresa?"
    print(abaco_client(pregunta_usuario_creditos))
