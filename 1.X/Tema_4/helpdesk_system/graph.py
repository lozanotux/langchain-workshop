import sqlite3
from operator import add
from typing import Annotated, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from config import *
from prompts import *
from rag_system import VectorRAGSystem


# Definicion del Estado
class HelpdeskState(TypedDict):
    consulta: str
    categoria: str  # "automatica" o "escalada"
    respuesta_rag: Optional[str]
    confianza: float
    fuentes: List[str]
    contexto_rag: Optional[str]
    requiere_humano: bool
    respuesta_humano: Optional[str]
    respuesta_final: Optional[str]
    historial: Annotated[List[str], add]


class HelpdeskGraph:
    """Grafo del sistema Helpdesk."""

    def __init__(self):
        load_dotenv()

        self.llm = ChatMistralAI(model=LLM_MODEL, temperature=0.1)
        self.rag = VectorRAGSystem(chroma_path=CHROMADB_PATH)
        self.graph = None

    def procesar_rag(self, state):
        """Busca el contexto de la consulta utilizando el sistema RAG."""
        consulta = state["consulta"]
        resultado = self.rag.buscar(consulta)
        return {
            "respuesta_rag": resultado["respuesta"],
            "confianza": resultado["confianza"],
            "fuentes": resultado["fuentes"],
            "contexto_rag": resultado["respuesta"],
            "historial": [
                f"RAG ejecutado con MultiQueryRetriever",
                f"Confianza: {resultado["confianza"]}",
                f"Fuentes consultadas: {len(resultado['fuentes'])}",
            ],
        }

    def clasificar_con_contexto(self, state):
        """Clasifica la consulta para responder automaticamente o escalar con el contexto del RAG."""
        consulta = state["consulta"]
        contexto_rag = state.get("contexto_rag", "")
        confianza = state.get("confianza", 0)

        prompt = ChatPromptTemplate.from_template(CLASIFICAR_CONSULTA_PROMPT)

        try:
            response = self.llm.invoke(
                prompt.format(
                    consulta=consulta, contexto_rag=contexto_rag, confianza=confianza
                )
            )

            content = response.content.strip().lower()

            if "automatico" in content or "automático" in content:
                categoria = "automatico"
            elif "escalado" in content:
                categoria = "escalado"
            else:
                categoria = "automatico" if confianza >= 0.60 else "escalado"

            return {
                "categoria": categoria,
                "historial": [
                    f"Clasificación con contexto: {categoria}",
                    f"Justificación: {response.content}",
                ],
            }
        except Exception as e:
            categoria = "automatico" if confianza >= 0.60 else "escalado"
            return {
                "categoria": categoria,
                "historial": [
                    f"Error en la clasificación, usando confianza: {confianza}"
                ],
            }

    def preparar_escalado(self, state):
        """Preaparar el escalado a un humano."""
        return {
            "requiere_humano": True,
            "historial": ["Escalado a agente humano - esperando intervención."],
        }

    def procesar_respuesta_humano(self, state):
        """Procesa la respueta del humano."""
        respuesta_humano = state.get("respuesta_humano", "")

        if respuesta_humano:
            return {
                "respuesta_final": respuesta_humano,
                "historial": ["Agente humano proporcionó respuesta."],
            }

        return {"historial": ["Esperando respuesta del agente humano"]}

    def generar_respuesta_final(self, state):
        """Genera la respueta final del sistema al ticket del usuario."""
        if state.get("respuesta_final"):
            return {"historial": ["Respuesta final proporcionada por agente humano."]}

        # Si no hay respueta final, la generamos con IA (usamos la respueta del sistema RAG)
        respuesta_rag = state.get("respuesta_rag", "")
        fuentes = state.get("fuentes", [])

        # Enriquecer respuesta final
        respuesta_final = respuesta_rag
        if fuentes:
            fuentes_texto = ", ".join(fuentes)
            respuesta_final += f"\n\nFuentes consultadas: {fuentes_texto}"

        return {
            "respuesta_final": respuesta_final,
            "historial": ["Respuesta final generada automaticamente."],
        }

    # Funciones de enrutamiento
    def decidir_desde_clasificacion(self, state):
        """Decide hacia donde ir despues de la clasificacion con contexto RAG."""
        categoria = state.get("categoria", "escalado")
        if categoria == "automatico":
            return "respuesta_final"
        else:
            return "escalado"

    def decidir_desde_humano(self, state):
        """Decide si continuar o esperar respuesta humana."""
        respuesta_humano = state.get("respuesta_humano", "")

        if respuesta_humano:
            return "procesar_humano"
        else:
            return "esperar"

    def crear_grafo(self):
        """Crear el grafo de LangGraph con los nodos y control de flujo."""
        graph = StateGraph(HelpdeskState)

        # Agregar nodos
        graph.add_node("rag", self.procesar_rag)
        graph.add_node("clasificar", self.clasificar_con_contexto)
        graph.add_node("escalado", self.preparar_escalado)
        graph.add_node("respuesta_final", self.generar_respuesta_final)
        graph.add_node("procesar_humano", self.procesar_respuesta_humano)

        # Definir la estructura del grafo
        graph.add_edge(START, "rag")
        graph.add_edge("rag", "clasificar")

        # Edges condicionales del grafo
        graph.add_conditional_edges(
            "clasificar",
            self.decidir_desde_clasificacion,
            {
                "respuesta_final": "respuesta_final",
                "escalado": "escalado"
            }
        )

        graph.add_conditional_edges(
            "escalado",
            self.decidir_desde_humano,
            {
                "procesar_humano": "procesar_humano",
                "esperar": END,  # Pausar la ejecucion del grafo hasta que responda el humano
            },
        )

        graph.add_edge("procesar_humano", END)
        graph.add_edge("respuesta_final", END)

        self.graph = graph
        return graph

    def compilar(self):
        """Compila el grafo con checkpointer."""
        if not self.graph:
            self.crear_grafo()

        conn = sqlite3.connect("helpdesk.db", check_same_thread=False)

        checkpointer = SqliteSaver(conn)

        compiled = self.graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["procesar_humano"]
        )

        return compiled


def crear_helpdesk():
    helpdesk = HelpdeskGraph()
    return helpdesk.compilar()
