import warnings

warnings.filterwarnings(
    "ignore",
    message=".*Pydantic V1 functionality.*",
    category=UserWarning,
    module="langchain_core.*",
)

import json
import re

from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_mistralai import ChatMistralAI

load_dotenv()

# Configuración del modelo
llm = ChatMistralAI(model="mistral-medium-latest", temperature=0)


def get_json(text: str) -> dict | None:
    """Extrae el bloque JSON del texto dado."""
    match = re.search(r"```json\s*(.+?)\s*```", text, re.DOTALL)
    if not match:
        return text

    return match.group(1).strip()


def preprocesar_texto(texto):
    """Limpia el texto elimninando espacios extras y limitando longitud."""
    return texto.strip()[:500]


# Convertir la función en un Runnable
preprocesador = RunnableLambda(preprocesar_texto)


def generar_resumen(texto):
    """Genera un resumen conciso del texto"""
    prompt = f"Resume en una sola oración: {texto}"
    respuesta = llm.invoke(prompt)
    return respuesta.content


rama_resumen = RunnableLambda(generar_resumen)


def analizar_sentimiento(texto):
    """Analiza el sentimiento y devuelve resultado estructurado."""
    prompt = f"""Analiza el sentimiento del siguiente texto.
    Responde UNICAMENTE con un JSON string válido (no lo hagas en formato markdown, evita los backsticks):
    {{"sentimiento"": "positivo|negativo|neturo", "razon": "justificacion breve"}}

    Texto: {texto}"""
    respuesta = llm.invoke(prompt)
    try:
        #return json.loads(get_json(respuesta.content)) # la funcion get_json nacio de un mal prompt
        return json.loads(respuesta.content)
    except json.JSONDecodeError:
        return {"sentimiento": "neutro", "razon": "Error en análisis"}


rama_sentimiento = RunnableLambda(analizar_sentimiento)


def combinar_resultados(datos):
    """Combina los resultados de ambas ramas en un formato unificado."""
    return {
        "resumen": datos["resumen"],
        "sentimiento": datos["sentimiento_data"]["sentimiento"],
        "razon": datos["sentimiento_data"]["razon"],
    }


combinador = RunnableLambda(combinar_resultados)


# def proceso_uno(t):
#     resumen = generar_resumen(t)
#     sentimiento_data = analizar_sentimiento(t)
#     return combinar_resultados(
#         {
#             "resumen": resumen,
#             "sentimiento_data": sentimiento_data,
#         }
#     )


# # Convertir en Runnable
# proceso = RunnableLambda(proceso_uno)

analizador_paralelo = RunnableParallel(
    {
        "resumen": rama_resumen,
        "sentimiento_data": rama_sentimiento,
    }
)

# La cadena completa
# cadena = preprocesador | proceso
cadena = preprocesador | analizador_paralelo | combinador

# Prueba con diferentes textos
textos_prueba = [
    "¡Me encanta este producto! Funciona perfectamente y llegó muy rápido.",
    "El servicio al cliente fue terrible, nadie me ayudó con mi problema.",
    "El clima está nublado hoy, probablemente llueva más tarde.",
]

# Secuenialmente
# for texto in textos_prueba:
#     resultado = cadena.invoke(texto)
#     print(f"Texto: {texto}")
#     print(f"Resultado: {resultado}")
#     print("-" * 80)

# Batch (paralelamente)
resultados = cadena.batch(textos_prueba)

for resultado in resultados:
    print(f"Resultado: {resultado}")
    print("-" * 80)