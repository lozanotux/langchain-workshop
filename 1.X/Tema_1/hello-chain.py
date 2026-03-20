from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from dotenv import load_dotenv


load_dotenv()

chat = ChatMistralAI(model="mistral-large-latest", temperature=0.7)

plantilla = PromptTemplate(
    input_variables=["nombre"],
    template=(
        "Saluda al usuario con su nombre.\n" \
        "Nombre del usuario: {nombre}\n" \
        "Asistente:"
    )
)

cadena = plantilla | chat

respuesta = cadena.invoke({"nombre": "Juan"})
print(respuesta.content)