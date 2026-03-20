from langchain_core.tools import StructuredTool


def herramienta_estructurada(query: str) -> str:
    """Herramienta estructurada de consulta de usuarios"""
    # Realmente se implementaria codigo para acceder a una base
    # de datos y hacer una query SQL para traerse los usuarios.
    return f"Usuario: {query}"


mi_tool = StructuredTool.from_function(herramienta_estructurada)

print(mi_tool.run("Juan"))
