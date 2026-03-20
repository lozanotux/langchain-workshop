"""
Características clave de los Retrievers:
- Entrada: string de consulta
- Salida: lista de objetos Document
- Implementan la interfaz Runnable estándar
- Compatibles con LCEL (LangChain Expression Language)
"""
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv


load_dotenv()

vector_store = Chroma(
    embedding_function=MistralAIEmbeddings(model="mistral-embed"),
    persist_directory="./Tema_3/vector_store"
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos?"
resultados = retriever.invoke(consulta)

print("Top 2 documentos mas similares a la consulta:")
for i, doc in enumerate(resultados):
    print(f"Documento {i+1}:\n{doc.page_content}\n")
    print(f"Metadata: {doc.metadata}\n")
