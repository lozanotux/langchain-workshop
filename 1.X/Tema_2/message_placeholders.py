from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un experto en análisis de sentimientos. Clasifica cada texto como POSITIVO, NEGATIVO o NEUTRO."),
    MessagesPlaceholder(variable_name="ejemplos"),
    ("human", "Texto a analizar: {texto_usuario}"),
])

# Simulamos un historial de conversación
ejemplos_sentimientos = [
    HumanMessage(content="Texto a analizar: Me encanta este producto, es increíble"),
    AIMessage(content="POSITIVO"),
    HumanMessage(content="Texto a analizar: Este servicio es terrible, no lo recomiendo"),
    AIMessage(content="NEGATIVO"),
    HumanMessage(content="Texto a analizar: El clima está nublado hoy"),
    AIMessage(content="NEUTRO")
]

mensajes = chat_prompt.format_messages(
    ejemplos=ejemplos_sentimientos,
    texto_usuario="Me encanta este nuevo restaurante, la comida es deliciosa"
)

for i, m in enumerate(mensajes):
    print(f"Mensaje {i}: {type(m).__name__}: {m.content}")
    print("-" * 50)

# Few-shot prompting con el modelo de lenguaje
from langchain_mistralai import ChatMistralAI

chat_model = ChatMistralAI(model="mistral-medium-latest", temperature=0)

respuesta = chat_model.invoke(mensajes)

print(respuesta.content)
