"""
Casos de uso ideales:
 - Manuales técnicos
 - Contratos y documentos legales
 - Papers académicos
 - Reportes empresariales
"""
# PDF LOADER

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("file.pdf")

pages = loader.load()

for i, page in enumerate(pages):
    print(f"=== Page {i + 1} ===")
    print(page.page_content)
    print(page.metadata)


# WEB LOADER

from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.example.com")

docs = loader.load()

for i, doc in enumerate(docs):
    print(f"=== Document {i + 1} ===")
    print(doc.page_content)
    print(doc.metadata)

# Directory Loader
"""
Casos de uso ideales:
 - Bases de conocimiento empresariales
 - Repositorios de código con documentación
 - Colecciones de artículos
 - Archivos de proyectos completos
"""

# YouTube Loader
"""
Casos de uso ideales:
 - Análisis de contenido educativo
 - Transcripción de conferencias
 - Análisis de tendencias en videos
 - Creación de resúmenes automáticos
"""

# UnstructuredHTMLLoader
"""
Casos de uso ideales:
 - Procesamiento de reportes web
 - Análisis de documentación HTML
 - Extracción de contenido de emails HTML
 - Procesamiento de archivos exportados
"""

# CSV Loader
"""
Casos de uso ideales:
 - Análisis de datos de ventas
 - Logs de sistema
 - Datos de encuestas
 - Registros de transacciones
"""

# GitLoader
"""
Casos de uso ideales:
 - Análisis de código fuente
 - Documentación de proyectos
 - Auditorías de repositorios
 - Generación de documentación automática
"""