from config import config
from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from prompts import *
from tools import gmail_tools, search_tool, virustotal_checker

# Inicializar LLM
llm = ChatMistralAI(
    model="mistral-large-latest", api_key=config.MISTRAL_API_KEY, temperature=0.1
)

# Agente 1: Analisis de Alertas
alert_analyzer = create_agent(
    model=llm,
    tools=[search_tool, virustotal_checker],
    system_prompt=ALERT_PROMPT,
    name="alert_analyzer",
)

# Agente 2: Analisis de Amenazas y Mitigaciones
threat_analyzer = create_agent(
    model=llm,
    tools=[search_tool],
    system_prompt=THREAT_PROMPT,
    name="threat_analyzer",
)

# Agente 3: Notificaciones
notification_agent = create_agent(
    model=llm,
    tools=gmail_tools,
    system_prompt=NOTIFICATION_PROMPT,
    name="notification_agent",
)
