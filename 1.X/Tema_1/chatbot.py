import warnings

warnings.filterwarnings(
    "ignore",
    message=".*Pydantic V1 functionality.*",
    category=UserWarning,
    module="langchain_core.*",
)

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI

load_dotenv()

# Configurar la pagina de la app
st.set_page_config(page_title="Chatbot Básico", page_icon="🤖")
st.title("🤖  Chatbot Básico con LangChain")
st.markdown("""
    #### Es un *ejemplo* construido con LangChain + Streamlit.

    ¡Escribe un mensaje para comenzar!
    """)

with st.sidebar:
    if st.button("🗑️ Nueva conversación"):
        st.session_state.mensajes = []
        st.rerun()

    st.header("Configuración")
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.5, 0.1)

    opciones = {
        "Mistral Large": "mistral-large-latest",
        "Mistral Medium": "mistral-medium-latest",
        "Codestral": "codestral-latest",
    }
    model_name = st.selectbox("Modelo", opciones.keys())

    chat_model = ChatMistralAI(model=opciones[model_name], temperature=temperature)

prompt_template = PromptTemplate(
    input_variables=["mensaje", "historial"],
    template="""Eres un asistente útil y amigable llamado ChatBot Pro. 
 
Historial de conversación:
{historial}
 
Responde de manera clara y concisa a la siguiente pregunta: {mensaje}""",
)

# En lugar de usar cadenas tradicionales, ahora puedes hacer:
cadena = prompt_template | chat_model

# Inicializar el historial de mensajes
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Mostrar mensajes previos en la interfaz
for msg in st.session_state.mensajes:
    if isinstance(msg, SystemMessage):
        continue  # No mostrar mensajes del sistema

    role = "assistant" if isinstance(msg, AIMessage) else "user"

    with st.chat_message(role):
        st.markdown(msg.content)

# Cuadro de entrada de texto de usuario
pregunta = st.chat_input("Escribe tu mensaje: ")

if pregunta:
    with st.chat_message("user"):
        st.markdown(pregunta)

    try:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            # ¡Aquí está la magia del streaming!
            for chunk in cadena.stream(
                {"mensaje": pregunta, "historial": st.session_state.mensajes}
            ):
                full_response += chunk.content
                response_placeholder.markdown(
                    full_response + "▌"
                )  # El cursor parpadeante

            response_placeholder.markdown(full_response)

        # No olvides almacenar los mensajes
        st.session_state.mensajes.append(HumanMessage(content=pregunta))
        st.session_state.mensajes.append(AIMessage(content=full_response))

    except Exception as e:
        st.error(f"Error al generar respuesta: {str(e)}")
        st.info("Verifica que tu API Key de Mistral esté configurada correctamente.")
