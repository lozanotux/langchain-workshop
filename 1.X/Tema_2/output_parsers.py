from pydantic import BaseModel, Field
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()

class AnalisisTexto(BaseModel):
    resumen: str = Field(description="Resumen breve del texto.")
    sentimiento: str = Field(description="Sentimiento del texto (positivo, negativo, neutro).")

llm = ChatMistralAI(model="mistral-medium-latest", temperature=0.6)

structured_llm = llm.with_structured_output(AnalisisTexto)

texto_prueba = "Me encantó la nueva pelicula de acción, tiene muchos efectos especiales impresionantes y acción."

resultado = structured_llm.invoke(f"Analiza el siguiente texto: {texto_prueba}")

# print(type(resultado))
# print(dir(resultado))

print(resultado.model_dump_json(indent=2))
