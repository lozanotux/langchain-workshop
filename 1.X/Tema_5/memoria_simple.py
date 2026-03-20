from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mistralai import ChatMistralAI
from langchain_core.runnables.history import RunnableWithMessageHistory  # Se guarda en RAM
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

llm = ChatMistralAI(model="mistral-large-latest", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil"),
    # La gestion de recursos puede ser elevados con este metodo
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# Cadena con memoria automatica por sesion
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

print("Chat en terminal (escribe 'salir' para terminar)\n")
session_id = "sesion_terminal"

while True:
    try:
        user_input = input("Tú: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nHasta luego!")
        break

    if not user_input:
        continue

    if user_input.lower() in {"salir", "exit", "quit"}:
        print("\nHasta luego!")
        break

    respuesta = chain_with_memory.invoke(
        {"input": user_input},
        config={
            "configurable": {
                "session_id": session_id
            }
        }
    )
    print("Asistente:", respuesta.content)
