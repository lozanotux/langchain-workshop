"""
Este script necesita de un archivo 'credentials.json' de tu cuenta
de Google con los permisos necesarios para poder interactuar con la
toolkit de GMail.
"""
import base64
import os
from email.mime.text import MIMEText

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.agent_toolkits import GmailToolkit
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# Configurar directorio de trabajo
orig_dir = os.getcwd()
os.chdir(os.path.join(orig_dir, "Tema_6"))


@tool
def create_gmail_reply_draft(
    message: str, to: str, subject: str, thread_id: str, in_reply_to: str = None
) -> str:
    """
    Crea un borrador de respuesta en GMail que SÍ funciona como respuesta en el hilo.
    Usa esta herramienta cuando quieras crear una RESPUESTA a un email existente.

    Args:
        message: contenido del mensaje de respuesta
        to: email del destinatario
        subject: asunto (deberia emprezar con Re:)
        thread_id: ID del hilo para que sea una respuesta (OBLIGATORIO)
        in_reply_to: message-ID del email original (opcional)
    """
    try:
        # Obtener service de GMail del toolkit
        gmail_toolkit = GmailToolkit()
        service = gmail_toolkit.api_resource

        # Crear mensaje MIME con headers de respuesta
        mime_message = MIMEText(message, "plain", "utf-8")
        mime_message["To"] = to
        mime_message["Subject"] = subject

        # Headers para que sea una respuesta
        if in_reply_to:
            mime_message["In-Reply-To"] = in_reply_to
            mime_message["References"] = in_reply_to

        # Codificar mensaje
        encoded_message = base64.urlsafe_b64encode(mime_message.as_bytes()).decode(
            "utf-8"
        )

        # Crear borrador con thread_id
        draft_body = {
            "message": {
                "raw": encoded_message,
                "threadId": thread_id,  # ¡Esto es lo critico!
            }
        }

        draft = service.users().drafts().create(userId="me", body=draft_body).execute()

        return f"Borrador de RESPUESTA creado exitosamente. Draft ID: {draft['id']}, Thread ID: {thread_id}"
    except Exception as e:
        return f"Error creando borrador de respuesta: {str(e)}"


# Configurar el toolkit de GMail
gmail_toolkit = GmailToolkit()
tools = gmail_toolkit.get_tools()
tools.append(create_gmail_reply_draft)

# Configurar modelo del agente que soporte tool calling
load_dotenv()
model = init_chat_model("mistralai:mistral-large-latest", temperature=0)

# Prompt de agente que define su comportamiento
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un asistente de email profesional. Para procesar emails sigue EXACTAMENTE estos pasos:

    1. PRIMERO: Usa 'search_gmail' con query 'in:inbox' para obtener la lista de mensajes en la bandeja de entrada.
    
    2. SEGUNDO: De la lista obtenida, identifica el message_id del email más reciente (el primer resultado).
    
    3. TERCERO: Usa 'get_gmail_message' con el message_id real obtenido en el paso anterior para obtener el contenido completo.
    
    4. CUARTO: Analiza el email y EXTRAE esta información crítica:
       - Thread ID (busca "Thread ID:" en el contenido)
       - Remitente original (busca "From:" y extrae el email)
       - Asunto original (busca "Subject:")
       - Contenido principal del mensaje
    
    5. QUINTO: Genera una respuesta profesional y apropiada en español.
    
    6. SEXTO: Usa 'create_gmail_draft' para crear un borrador de RESPUESTA (no email nuevo) con:
       - "message": tu respuesta generada
       - "subject": "Re: [asunto original]" (si no empieza ya con "Re:")
       - "to": email del remitente original
       - "thread_id": el Thread ID extraído del paso 4 (MUY IMPORTANTE para que sea una respuesta)

    CRÍTICO PARA RESPUESTAS:
    - SIEMPRE incluye "thread_id" en create_gmail_draft para que sea una respuesta, no un email nuevo
    - El "to" debe ser el email del remitente original
    - El "subject" debe empezar con "Re:" si no lo tiene ya

    IMPORTANTE: 
    - NUNCA uses message_id hardcodeados como '1' o '2' 
    - SIEMPRE obtén los IDs reales de los mensajes primero
    - Sin thread_id, el borrador será un email nuevo, no una respuesta
    - Si no encuentras thread_id, informa el problema pero intenta crear el borrador igual
    
    Si encuentras errores, explica qué información falta y por qué.""",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Crear agente
agent = create_tool_calling_agent(model, tools, prompt)

# Crear executor del agente
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,  # Limitar las iteraciones para evitar bucles *loops
)


def procesar_ultimo_email():
    try:
        response = agent_executor.invoke(
            {
                "input": "Procesa el email mas reciente en la bandeja de entrada y genera un borrador de respuesta profesional"
            }
        )
        return response["output"]
    except Exception as e:
        print(f"Error al procesar email: {e}")
        return f"Error {str(e)}"


# Ejecutar
if __name__ == "__main__":
    result = procesar_ultimo_email()
    print("\n" + "=" * 50)
    print("RESULTADO:")
    print("=" * 50)
    print(result)
