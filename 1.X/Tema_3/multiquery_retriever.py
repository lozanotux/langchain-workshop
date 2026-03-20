"""
El MultiQueryRetriever aborda las limitaciones de la búsqueda por similitud
basada en distancia generando múltiples "perspectivas" alternativas de tu
consulta original. En lugar de hacer una sola búsqueda, utiliza un LLM para
crear varias versiones de la misma pregunta y luego combina los resultados.

¿Cuándo usarlo?
- Cuando tu consulta original puede ser interpretada de múltiples formas
- Para mejorar la diversidad de resultados recuperados
- En casos donde la consulta inicial podría no capturar todos los aspectos relevantes
"""
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_classic.retrievers import MultiQueryRetriever
from dotenv import load_dotenv


load_dotenv()

vector_store = Chroma(
    embedding_function=MistralAIEmbeddings(model="mistral-embed"),
    persist_directory="./Tema_3/vector_store"
)

retriever_base = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)
retriever_avanzado = MultiQueryRetriever.from_llm(
    retriever=retriever_base,
    llm=ChatMistralAI(model="mistral-large-latest")
)

consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos?"
resultados = retriever_avanzado.invoke(consulta)

print("Top documentos mas similares a la consulta:")
for i, doc in enumerate(resultados):
    print(f"Documento {i+1}:\n{doc.page_content}\n")
    print(f"Metadata: {doc.metadata}\n")

"""
Otros Retrievers interesantes:
1. ContextualCompressionRetriever (El Filtro Inteligente): Comprime el contexto antes de la recuperación.
Beneficios
- Reduce costos de llamadas a LLM al eliminar texto irrelevante
- Mejora la calidad de las respuestas al proporcionar contexto más preciso
- Permite pasar más documentos relevantes dentro del límite de tokens

2. EnsembleRetriever (El Mejor de Dos Mundos): Combina múltiples retrievers para mejorar la precisión.
Por qué es efectivo
- BM25: Excelente para coincidencias exactas de palabras clave
- Vector Search: Superior para similitud semántica
- Fusión: Combina las fortalezas de ambos enfoques

Casos de uso ideales
- Búsquedas que requieren tanto precisión léxica como semántica
- Documentos técnicos con terminología específica
- Sistemas de búsqueda empresarial

3. ParentDocumentRetriever (Precisión con Contexto): Recupera documentos padres para proporcionar contexto adicional.
Ventajas clave
- Embeddings precisos: Los chunks pequeños crean embeddings más representativos
- Contexto completo: Devuelve documentos padre con contexto suficiente
- Flexibilidad: Puedes ajustar el tamaño de chunks padre e hijo independientemente

4. SelfQueryRetriever (Búsqueda Estructurada Inteligente): Utiliza un LLM para convertir consultas naturales en consultas estructuradas.
Casos de uso
- Bases de datos con metadatos ricos
- Consultas que combinan contenido y filtros
- Sistemas que requieren búsqueda estructurada automática

5. TimeWeightedVectorStoreRetriever (Memoria que Desvanece): Da más peso a documentos recientes en la recuperación.
Para información sensible al tiempo
- Este retriever asigna mayor importancia a documentos más recientes, simulando
cómo funciona la memoria humana.

6. Tecnicas avanzadas y combinaciones:
- Retrieval con Reranking (El Refinador): Después de recuperar documentos, utiliza un LLM para reordenarlos según su relevancia.
- Retrieval MMR (Maximum Marginal Relevance): Combina relevancia y diversidad para evitar resultados redundantes.
"""
