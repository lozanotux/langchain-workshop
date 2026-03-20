import os
from tkinter import Tk, filedialog
from typing import List, TypedDict, Annotated

from dotenv import load_dotenv
import httpx
from langchain_mistralai import ChatMistralAI
from langgraph.graph import END, START, StateGraph
from mistralai import Mistral
from moviepy import VideoFileClip
from operator import add

# Configuración
load_dotenv()
llm = ChatMistralAI(model="magistral-small-latest", temperature=0.3)


# Definicion del estado
class State(TypedDict):
    notes: str
    participants: List[str]
    topics: List[str]
    action_items: List[str]
    minutes: str
    summary: str
    logs: Annotated[List[str], add]


# ================= NODOS DEL WORKFLOW =================
def extract_participants(state: State) -> State:
    # Este prompt se puede hacer con pydantic para asegurarnos de que el formato de salida es correcto,
    # pero por simplicidad lo hacemos directo.
    prompt = f"""
    Analiza las siguientes notas de reunión y extrae únicamente los nombres de los participantes.
    
    Notas: {state['notes']}
    
    Instrucciones:
    - Responde SOLO con nombres separados por comas
    - No incluyas explicaciones adicionales
    - Formato: Juan García, María López, Carlos Ruiz
    """

    response = llm.invoke(prompt)
    participants = [p.strip() for p in response.text.split(",") if p.strip()]

    print(f"✓ Participantes extraídos: {len(participants)}")

    return {
        "participants": participants,
        "logs": ["Paso 1: Participantes extraídos [✓]"],
    }


def identify_topics(state: State) -> State:
    """Identifica los temas principales discutidos."""
    prompt = f"""
    Identifica los 3-5 temas principales discutidos en esta reunión.
    
    Notas: {state['notes']}
    
    Responde SOLO con los temas separados por punto y coma (;).
    Ejemplo: Arquitectura del sistema; Planificación del proyecto; Asignación de tareas
    """

    response = llm.invoke(prompt)
    topics = [t.strip() for t in response.text.split(";") if t.strip()]

    print(f"✓ Temas identificados: {len(topics)}")

    return {
        "topics": topics,
        "logs": ["Paso 2: Temas identificados [✓]"],
    }


def extract_action_items(state: State) -> State:
    """Extrae las acciones acordadas y sus responsables."""
    prompt = f"""
    Extrae las acciones especificas acordadas durante la reunión, incluyendo el responsable si se menciona.
    
    Notas: {state['notes']}
    
    Frormato de respuesta: Una acción por linea, separadas por |
    Ejemplo: María se encargará del backend | Juan revisará el diseño | Carlos coordinará la próxima reunión

    Si no hay acciones claras, responde con: "No se identificaron acciones específicas"
    """

    response = llm.invoke(prompt)

    if "No se identificaron acciones específicas" in response.content:
        action_items = []
    else:
        action_items = [item.strip() for item in response.text.split("|") if item.strip()]

    print(f"✓ Acciones extraídas: {len(action_items)} items")

    return {
        "action_items": action_items,
        "logs": ["Paso 3: Acciones extraídas [✓]"],
    }


def generate_minutes(state: State) -> State:
    """Genera una minuta formal de la reunión."""
    participants_str = ", ".join(state["participants"])
    topics_str = "\n• ".join(state["topics"])
    actions_str = (
        "\n• ".join(state["action_items"])
        if state["action_items"]
        else "No se identificaron acciones específicas"
    )

    prompt = f"""
    Genera una minuta formal y profesional basándote en la siguiente información:
    
    PARTICIPANTES: {participants_str}

    TEMAS DISCUTIDOS:
    • {topics_str}

    ACCIONES ACORDADAS:
    • {actions_str}

    NOTAS ORIGINALES: {state['notes']}
    
    Genera una minuta profesional de máximo 150 palabras que incluya:
    1. Encabezado con tipo de reunión
    2. Lista de asistentes
    3. Puntos principales discutidos
    4. Acciones y próximos pasos

    Usa un tono formal y estructura clara.
    """

    response = llm.invoke(prompt)

    print(f"✓ Minuta generada: {len(response.text.split())} palabras")

    return {"minutes": response.text}


def create_summary(state: State) -> State:
    """Crea un resumen ejecutivo ultra-breve."""
    prompt = f"""
    Crea un resumen ejecutivo de MÁXIMO 2 líneas (30 palabras) que capture la esencia de esta reunión.
    
    Participantes: {", ".join(state['participants'][:3])}{'...' if len(state['participants']) > 3 else ''}
    Temas clave: {', '.join(state['topics'][:3])}{'...' if len(state['topics']) > 3 else ''}
    Acciones clave: {len(state['action_items'])} acciones definidas
    
    El resumen debe ser conciso y directo al punto.
    """

    response = llm.invoke(prompt)

    print(f"✓ Resumen creado")

    return {"summary": response.text}


# ================= CONSTRUCCIÓN DEL GRAFO =================
def create_workflow():
    """Crea y configura el worklfow de LangGraph."""
    workflow = StateGraph(State)

    # Agrega todos los nodos
    workflow.add_node("extract_participants", extract_participants)
    workflow.add_node("identify_topics", identify_topics)
    workflow.add_node("extract_action_items", extract_action_items)
    workflow.add_node("generate_minutes", generate_minutes)
    workflow.add_node("create_summary", create_summary)

    # Configurar flujo secuencial
    workflow.add_edge(START, "extract_participants")
    workflow.add_edge("extract_participants", "identify_topics")
    workflow.add_edge("identify_topics", "extract_action_items")
    workflow.add_edge("extract_action_items", "generate_minutes")
    workflow.add_edge("generate_minutes", "create_summary")
    workflow.add_edge("create_summary", END)

    return workflow.compile()


# ================= FUNCIONES DE PROCESAMIENTO ==================
def transcribe_media_direct(file_path: str) -> str:
    """
    Transcribe usando directamente la API de MistralAI Voxtral.
    Voxtral solo puede procesar archivos de audio, así que si el archivo es un video,
    primero extraemos el audio.
    INFO: https://docs.mistral.ai/capabilities/audio_transcription/offline_transcription
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    model = "voxtral-mini-latest"
    audio_path = file_path

    try:
        print("🎙️  Transcribiendo con MistralAI Voxtral API directa...")

        client = Mistral(api_key=api_key, client=httpx.Client(verify=False))

        # 1. Extraer audio si el archivo original es un video
        if file_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            print("🎬 Video detectado. Extrayendo audio...")
            audio_path = "temp_audio.mp3"
            with VideoFileClip(file_path) as video:
                # logger=None evita que moviepy llene la consola con barras de carga (ideal para tutoriales)
                video.audio.write_audiofile(audio_path, logger=None)

        # 2. Subir el archivo de audio (sea el original o el temporal)
        print("☁️  Subiendo archivo a Mistral...")
        with open(audio_path, "rb") as audio_file:
            uploaded_audio = client.files.upload(
                file={"content": audio_file, "file_name": os.path.basename(audio_path)},
                purpose="audio",
            )
        signed_url = client.files.get_signed_url(file_id=uploaded_audio.id)

        # 3. Limpieza: Eliminar el audio temporal si fue creado
        if audio_path != file_path:
            os.remove(audio_path)

        # 4. Solicitar la transcripción
        print("📝 Procesando transcripción...\n")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": signed_url.url,
                    },
                    {
                        "type": "text",
                        "text": "Esta es una reunión de trabajo en español con múltiples participantes. Por favor, transcribe el audio.",
                    },
                ],
            }
        ]

        transcript = client.chat.complete(model=model, messages=messages)

        print("✓ Transcripción completada")
        return transcript.choices[0].message.content
    except Exception as e:
        print(f"Error durante la transcripción: {e}")
        raise


def process_meeting_notes(notes: str, app):
    """Procesa una nota de reunión individual."""
    initial_state = {
        "notes": notes,
        "participants": [],
        "topics": [],
        "action_items": [],
        "minutes": "",
        "summary": "",
        "logs": [],
    }

    print("\n" + "=" * 60)
    print("🔁 Procesando nota de reunión...")
    print("=" * 60)

    result = app.invoke(initial_state)
    return result


def display_results(result: State, meeting_num: int):
    """Muestra los resultados de forma estructurada."""
    print(f"\n📋 RESULTADOS - REUNIÓN #{meeting_num}")
    print("-" * 60)

    print(f"\n👥 Participantes ({len(result['participants'])}):")
    for p in result["participants"]:
        print(f"  • {p}")

    print(f"\n📌 Temas tratados ({len(result['topics'])}):")
    for t in result["topics"]:
        print(f"  • {t}")

    print(f"\n✅ Acciones acordadas ({len(result['action_items'])}):")
    if result["action_items"]:
        for a in result["action_items"]:
            print(f"  • {a}")
    else:
        print("  No se definieron acciones específicas.")

    print(f"\n📝 MINUTA FORMAL:")
    print("-" * 40)
    print(result["minutes"])
    print("-" * 40)

    print(f"\n📄 RESUMEN EJECUTIVO:")
    print(result["summary"])

    print("\n" + "=" * 60)

    print("\n📊 LOGS DE PROCESAMIENTO:")
    print(result["logs"])


# ================= DEMOSTRACIÓN ==================
if __name__ == "__main__":
    app = create_workflow()

    # Pequeña intefaz gráfica: selector de archivo
    Tk().withdraw()  # Oculta la ventana principal de Tkinter
    file_path = filedialog.askopenfilename(
        title="Selecciona un archivo de audio o video de reunión",
        filetypes=[
            ("Audio/Video Files", "*.mp4 *.mov *.mkv *.avi *.mp3 *.wav"),
            ("Text Files", "*.txt *.md"),
        ],
    )

    if not file_path:
        print("No se seleccionó ningún archivo.")
        raise SystemExit(0)

    ext = os.path.splitext(file_path)[1].lower()
    media_exts = {".mp4", ".mov", ".mkv", ".avi", ".mp3", ".wav"}

    if ext in media_exts:
        # Procesar con transcripción directa
        notes = transcribe_media_direct(file_path)
    else:
        # Si es un archivo de texto, lo leemos directamente
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            notes = f.read()

    result = process_meeting_notes(notes, app)
    display_results(result, meeting_num=1)
