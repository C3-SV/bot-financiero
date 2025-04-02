# SI ES CLIENTE DE ABACO 
# ESTO SERA MEJORADO LUEGO, PARA PROTOTIPO SERA BOOL 
from is_client import get_is_abaco_client
es_cliente_abaco = get_is_abaco_client()
print(f"From active_tools, es cliente abaco = {es_cliente_abaco}")
                   
# TOOLS
#? Tools generales
tools_not_client = [] 

# PREGUNTAS FINANCIERAS 
import tools.general.preguntas_tool as preguntas_tool
tool_preguntas = preguntas_tool.tool_preguntas
tools_not_client.append(tool_preguntas)

#* Tools de la plataforma de Abaco
from tools.abaco_platform.abaco_client_tool import tool_abaco_client

#! CONFIGURAR AL AGENTE 
# Lista de herramientas para el agente
def get_tools():
    global tools_abaco, tools_not_client
    es_cliente_abaco = get_is_abaco_client()
    agent_tools = []
    if es_cliente_abaco:
        agent_tools.append(tool_abaco_client)
    else:
        agent_tools.append(tool_preguntas)
    return agent_tools
