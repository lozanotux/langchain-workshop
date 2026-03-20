from langchain_core.tools import tool


@tool(
    "Herramienta de Acceso a Base de Datos de Usuarios",
    return_direct=True  # Evita que vuelva al LLM y responde directamente
)
def herramienta_personalizada(query: str) -> str:
    """Consulta la base de usuarios de la empresa"""
    # Codigo que accede a la base de datos
    return f"Respuesta a la consulta: {query}"


output = herramienta_personalizada.run("Consulta de prueba")
print(f"Nombre de la herramienta: {herramienta_personalizada.name}")
print(f"Descripcion de la herramienta: {herramienta_personalizada.description}")
print(f"Ejecucion de la herramineta:\n\n - {output} \n")
