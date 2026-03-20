import json
from datetime import datetime

from agents import alert_analyzer, notification_agent, threat_analyzer
from config import config
from langchain_mistralai import ChatMistralAI
from langgraph_supervisor import create_supervisor
from prompts import *

# Inicializar el modelo para el supervisor
supervisor_model = ChatMistralAI(
    model="mistral-large-latest", api_key=config.MISTRAL_API_KEY, temperature=0.1
)


def build_soc_workflow():

    # Crear el supervisor multiagente
    supervisor = create_supervisor(
        agents=[alert_analyzer, threat_analyzer, notification_agent],
        model=supervisor_model,
        prompt=SUPERVISOR_PROMPT,
        add_handoff_back_messages=True,
        output_mode="full_history",
    )

    return supervisor.compile()


# Instancia global del workflow del supervisor
soc_workflow = build_soc_workflow()


def process_security_alert(
    alert_data: dict, incident_id: str, processing_context: dict = None
) -> dict:
    if processing_context is None:
        processing_context = {}

    # Preparar el mensaje inicial/usuario para el supervisor
    initial_message = f"""ALERTA SOC PARA PROCESAMIENTO SECUENCIAL:

ID: {incident_id}
DATOS: {json.dumps(alert_data, indent=2)}
EMAIL: {processing_context.get('email_recipient', config.SOC_EMAIL_RECIPIENT)}

INSTRUCCIÓN CLARA: Ejecutar EXACTAMENTE estos 3 pasos:
1. alert_analyzer → análisis IOCs y determinar VERDADERO/FALSO POSITIVO
2. SI verdadero positivo → threat_analyzer → evaluación severidad y mitigación  
3. notification_agent → envío email final

TERMINAR después del paso 3. NO continuar."""

    print(f"🚀 Iniciando arquitectura supervisor para {incident_id}")
    print(
        "🤖 Supervisor coordinará: alert_analyzer → threat_analyzer → notification_agent"
    )
    print("🌐 Usando APIs reales: VirusTotal, Gmail, TavilySearch, AbuseIPDB")

    try:
        # Ejecutar el workflow del supervisor
        result = soc_workflow.invoke(
            {"messages": [{"role": "user", "content": initial_message}]}
        )

        # Extraer los resultados de cada agente del historial de mensajes
        analysis_result = _extract_agent_result(result, "alert_analyzer")
        threat_result = _extract_agent_result(result, "threat_analyzer")
        notification_result = _extract_agent_result(result, "notification_agent")

        # Determinar las herramientas utilizadas para los resultados
        tools_used = ["langgraph-supervisor", "create_supervisor"]

        if analysis_result and (
            "VIRUSTOTAL" in analysis_result or "VirusTotal" in analysis_result
        ):
            tools_used.append("VirusTotal API")
        if analysis_result or threat_result:
            tools_used.append("TavilySearch API")
        if notification_result and (
            "GMAIL" in notification_result or "Email" in notification_result
        ):
            tools_used.append("GMail API")

        final_result = {
            "incident_id": incident_id,
            "status": "completed",
            "analysis_result": analysis_result or "No analysis found",
            "threat_assessment": threat_result or "No threat assessment performed",
            "notification_sent": notification_result or "No notification sent",
            "timestamp": datetime.now().isoformat(),
            "tools_used": tools_used,
            "supervisor_architecture": True,
            "apis_real": True,
            "processing_context": processing_context,
            "full_conversation": result.get("messages", []),
        }

        print(f"✅ Arquitectura supervisor completada para {incident_id}")
        print(f"🛠️ Herramientas utilizadas: {', '.join(tools_used)}")

        return final_result

    except Exception as e:
        print(f"❌ Error en arquitectura supervisor: {str(e)}")

        error_result = {
            "incident_id": incident_id,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "supervisor_architecture": True,
            "apis_real": True,
        }

        return error_result


def _extract_agent_result(workflow_result: dict, agent_name: str) -> str:
    try:
        messages = workflow_result.get("messages", [])

        # Buscar los mensajes de un agente especifico
        agent_messages = []
        for message in messages:
            if hasattr(message, "content") and agent_name in str(message).lower():
                agent_messages.append(message.content)
            elif isinstance(message, dict):
                content = message.get("content", "")
                if agent_name in str(message).lower() or agent_name in content.lower():
                    agent_messages.append(content)

        # Retornar el ultimo mensaje del agente o concatenar si hay multiples
        if agent_messages:
            return (
                agent_messages[-1]
                if len(agent_messages) == 1
                else "\n\n".join(agent_messages)
            )

        return ""

    except Exception as e:
        print(f"Error extrayendo el resultado del agente: {agent_name}: {str(e)}")
        return ""
