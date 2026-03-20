from typing import TypedDict
from langgraph.graph import StateGraph, START, END


"""
Quiero saber si el numero es par o impar
"""
# Definir el estado
class State(TypedDict):
    numero: int
    resultado: str


grafo = StateGraph(State)

# Definir los nodos del workflow
def caso_par(state):
    return {"resultado": "PAR"}

def caso_impar(state):
    return {"resultado": "IMPAR"}

grafo.add_node("Par", caso_par)
grafo.add_node("Impar", caso_impar)

# Definir la función de routing para decidir la rama de ejecucion
def decidir_rama(state):
    if state["numero"] % 2 == 0:
        return "Par"
    else:
        return "Impar"


# Añadir el edge condicional al workflow
grafo.add_conditional_edges(START, decidir_rama)

# Conectar ambos casos al final
grafo.add_edge("Par", END)
grafo.add_edge("Impar", END)

compiled = grafo.compile()

# Probar el grafo con ejemplos
print(f"resultado de 3: {compiled.invoke({"numero": 3})["resultado"]}")