from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os


load_dotenv()

# Documentos PDF con contratos de locaciones inmobiliarias
loader = PyPDFDirectoryLoader("./Tema_3/contratos")
documentos = loader.load()

print(f"- Documentos cargados: {len(documentos)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=1000
)

documentos_divididos = text_splitter.split_documents(documentos)

print(f"- Documentos divididos: {len(documentos_divididos)}")

# Crear el vector store
vector_store = Chroma.from_documents(
    documentos_divididos,
    embedding=MistralAIEmbeddings(model="mistral-embed"),
    persist_directory="./Tema_3/chroma_db"
)

consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos?"

resultados = vector_store.similarity_search(consulta, k=2)

print("Top 3 documentos mas similares a la consulta:\n")
for i, doc in enumerate(resultados):
    print(f"Documento {i+1}:\n{doc.page_content}\n")
    print(f"Metadatos: {doc.metadata}\n")
    print("-" * 50)
