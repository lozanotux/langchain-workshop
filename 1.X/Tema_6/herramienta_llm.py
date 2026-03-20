from operator import attrgetter

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI

load_dotenv()
llm = ChatMistralAI(model="mistral-large-latest", temperature=0.2)


@tool("user_database_tool")
def herramienta_1(query: str) -> str:
    """
    Consulta la base de datos de usuarios de la empresa y
    vevuelve el resultado de la consulta
    """
    return f"Usuario estandar con permisos elevados"


llm_with_tools = llm.bind_tools([herramienta_1])

cadena_1 = llm_with_tools | attrgetter("tool_calls") | herramienta_1.map()
respuesta = cadena_1.invoke(
    "Busca la informacion que hay en la base de datos para el usuario Juan"
)

print(respuesta[0].content)
