from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un traductor de español a inglés muy preciso."),
    ("human", "{texto}"),
])

mensajes = chat_prompt.format_messages(texto="¿Cómo estás?")

for m in mensajes:
    print(f"{type(m)}: {m.content}")

chat_model = ChatMistralAI(model="mistral-medium-latest", temperature=0)

respuesta = chat_model.invoke(mensajes)

print(respuesta.content)
