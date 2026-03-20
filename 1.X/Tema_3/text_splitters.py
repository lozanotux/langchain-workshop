from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# 1. Cargar el documento PDF
loader = PyPDFLoader("./Tema_3/file.pdf")
pages = loader.load()

# 2. Dividir el texto en fragmentos manejables
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(pages)

# 3. Configurar el modelo de lenguaje
llm = ChatMistralAI(model="mistral-large-latest", temperature=0.2)
summaries = []

# 4. Resumir cada fragmento
i = 0
for chunk in chunks:
    if i > 10:  # Limitar a los primeros 10 fragmentos para evitar costos excesivos
        break
    respuesta = llm.invoke(f"Haz un resumen de los puntos mas importantes del siguiente texto: {chunk.page_content}")
    summaries.append(respuesta.content)
    i += 1

final_summary = llm.invoke(f"Combina y sintetiza estos resumenes en un resumen coherente y completo: {' '.join(summaries)}")
print(final_summary.content)