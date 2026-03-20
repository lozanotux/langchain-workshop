from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langgraph_supervisor import create_supervisor

load_dotenv()
model = ChatMistralAI(model="mistral-large-latest", temperature=0)


# Definir herramientas personalizadas
@tool
def buscar_web(query: str) -> str:
    """Buscar informacion en la web."""
    return f"Resultados de busqueda para: {query}"


@tool
def calcular(expresion: str) -> str:
    """Realizar calculos matematicos."""
    print(f"DEBUG: {expresion}")
    return f"Resultado {eval(expresion)}"


# Crear agentes especializados
agent_investigacion = create_agent(
    model=model,
    tools=[buscar_web],
    system_prompt="Eres un especialista en investigacion web.",
    name="investigador",
)

agente_matematico = create_agent(
    model=model,
    tools=[calcular],
    system_prompt="Eres un especialista en calculos matematicos. Usarás el modulo 'eval' de python para ejecutar expresiones, por lo tanto transforma palabras como 'pi' en su valor real (ej: 3.14)",
    name="matematico",
)

# Crear supervisor que coordina los agentes anteriores
supervisor_graph = create_supervisor(
    agents=[agente_matematico, agent_investigacion],
    model=model,
    prompt="Eres un supervisor que delega tareas a especialista segun el tipo de consulta.",
)

supervisor = supervisor_graph.compile()

# Uso del sistema multi-agente
messages = [
    HumanMessage(
        content="Busca informacion sobre pi y luego calcula su valor multiplicado por 2."
    )
]
messages = supervisor.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()
