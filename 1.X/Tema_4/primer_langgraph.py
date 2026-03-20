from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. Definir el esquema del estado
class Estado(TypedDict):
    texto_original: str
    texto_mayus: str
    longitud: int


# 2. Crear el grafo de estado
grafo = StateGraph(Estado)

# 3. Definir las funciones de los nodos
def poner_en_mayusculas(estado):
    texto = estado['texto_original']
    return {
        "texto_mayus": texto.upper()
    }


def contar_caracteres(estado):
    texto = estado['texto_mayus']
    return {
        "longitud": len(texto)
    }


# 4. Agregar nodos al grafo
grafo.add_node("Mayus", poner_en_mayusculas)
grafo.add_node("Contar", contar_caracteres)

# 5. Conectar los nodos en secuencia
grafo.add_edge(START, "Mayus")
grafo.add_edge("Mayus", "Contar")
grafo.add_edge("Contar", END)

# 6. Compilar el grafo
grafo_compilado = grafo.compile()

# 7. Invocar el grafo con un estado inicial
estado_inicial = {
    "texto_original": "Hola Mundo"
}
resultado = grafo_compilado.invoke(estado_inicial)
print(resultado)
