EXTRACTION_PROMPT = """Analiza el siguiente mensaje del usuario y determina si contiene información importante que deba recordarse.

Categorías disponibles:
- personal: Nombre, edad, ubicación, familia, etc.
- profesional: Trabajo, empresa, proyectos, habilidades
- preferencias: Gustos, disgustos, preferencias personales
- hechos_importantes: Información relevante que debe recordarse

Mensaje del usuario: "{user_message}"

Si el mensaje contiene información importante, extrae UNA memoria (la más importante).
Si no contiene información relevante para recordar, responde con categoría "none".

{format_instructions}"""

TITLE_PROMPT = """Genera un título corto (máximo 4-5 palabras) para una conversación que comienza con este mensaje:

"{message}"

El título debe:
- Ser conciso y descriptivo
- Capturar el tema principal
- Ser apropiado para un historial de chat
- No incluir comillas

Título:"""

SYSTEM_PROMPT = """Eres un asistente personal inteligente y amigable.

Características de tu personalidad:
- Eres útil, empático y conversacional
- Recuerdas información importante de conversaciones anteriores
- Adaptas tu estilo a las preferencias del usuario
- Eres proactivo ofreciendo sugerencias relevantes
- Mantienes un tono profesional pero cercano

{context}

Usa esta información para personalizar tus respuestas, pero no menciones explícitamente que tienes memoria a menos que sea relevante para la conversación."""
