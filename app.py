# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
import uuid
import shutil
from PyPDF2 import PdfReader
import docx
import requests
import json
from typing import List, Dict, Any, Optional
from decouple import config
import numpy as np
from sentence_transformers import SentenceTransformer

# Crear directorios necesarios
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

app = FastAPI(title="Chatbot de Documentos con IA")

# Montar directorio de archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de Deepseek API
DEEPSEEK_API_KEY = config("DEEPSEEK_API_KEY", default="")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Inicializar modelo de embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Almacenamiento en memoria para documentos
documents = {}
document_embeddings = {}

# Modelos para API
class Question(BaseModel):
    question: str
    document_id: str

class WidgetConfig(BaseModel):
    api_key: str
    theme: Optional[str] = "light"
    position: Optional[str] = "bottom-right"

class ChatSession(BaseModel):
    session_id: str
    document_id: str
    api_key: str

# Extraer texto de diferentes tipos de documentos
def extract_text(file_path):
    _, extension = os.path.splitext(file_path)
    
    if extension.lower() == '.pdf':
        text = ""
        with open(file_path, 'rb') as f:
            pdf = PdfReader(f)
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    
    elif extension.lower() == '.docx':
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    elif extension.lower() in ['.txt', '.csv', '.md']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    else:
        raise ValueError(f"Formato de archivo no soportado: {extension}")

# Procesar documento y crear embeddings
def process_document(text):
    # Dividir el texto en chunks significativos
    chunks = []
    paragraphs = text.split('\n')
    
    current_chunk = ""
    for paragraph in paragraphs:
        if len(paragraph.strip()) < 5:  # Saltar líneas muy cortas
            continue
            
        # Si el chunk actual + el párrafo nuevo es demasiado largo, guardar chunk actual y empezar uno nuevo
        if len(current_chunk) + len(paragraph) > 500:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk += " " + paragraph if current_chunk else paragraph
    
    # No olvidar el último chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Generar embeddings para cada chunk
    chunk_embeddings = embedding_model.encode(chunks)
    
    return chunks, chunk_embeddings

# Encontrar chunks relevantes para una pregunta
def find_relevant_chunks(chunks, chunk_embeddings, question, top_k=3):
    # Generar embedding para la pregunta
    question_embedding = embedding_model.encode([question])[0]
    
    # Calcular similitud de coseno entre la pregunta y cada chunk
    similarities = np.dot(chunk_embeddings, question_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(question_embedding)
    )
    
    # Obtener los top_k chunks más relevantes
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    relevant_chunks = [chunks[i] for i in top_indices]
    relevance_scores = [float(similarities[i]) for i in top_indices]
    
    return relevant_chunks, relevance_scores

# Función para consultar a Deepseek API
def query_deepseek(question, context_chunks):
    if not DEEPSEEK_API_KEY:
        return "Error: API key de Deepseek no configurada."
    
    # Construir contexto
    context = "\n\n".join(context_chunks)
    
    # Construir prompt
    prompt = f"""Actúa como un asistente experto que responde preguntas basadas en la información proporcionada.
    
CONTEXTO:
{context}

INSTRUCCIONES:
- Responde la pregunta basándote únicamente en la información del CONTEXTO proporcionado.
- Si la información no está en el CONTEXTO, indica honestamente que no puedes responder.
- Sé conciso y directo en tus respuestas.
- No inventes información.

PREGUNTA: {question}

RESPUESTA:"""

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }
        
        data = {
            "model": "deepseek-chat",  # Ajusta según la documentación de Deepseek
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,  # Baja temperatura para respuestas más precisas
            "max_tokens": 500
        }
        
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            data=json.dumps(data)
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            return answer
        else:
            return f"Error al consultar la API: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error al procesar la pregunta: {str(e)}"

# Página principal con HTML mejorado
@app.get("/", response_class=HTMLResponse)
async def get_home():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chatbot de Documentos con IA</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            body { 
                padding: 20px; 
                background-color: #f8f9fa;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .chat-container {
                height: 400px;
                overflow-y: auto;
                border: 1px solid #dee2e6;
                border-radius: 10px;
                padding: 15px;
                background-color: white;
                margin-bottom: 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .user-message {
                background-color: #e3f2fd;
                padding: 10px 15px;
                border-radius: 15px 15px 0 15px;
                margin-bottom: 15px;
                max-width: 80%;
                margin-left: auto;
                text-align: right;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            }
            .bot-message {
                background-color: #f1f1f1;
                padding: 10px 15px;
                border-radius: 15px 15px 15px 0;
                margin-bottom: 15px;
                max-width: 80%;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            }
            .bot-message.thinking {
                background-color: #e9ecef;
                color: #6c757d;
            }
            .navbar-brand {
                font-weight: bold;
                color: #0d6efd;
                font-size: 1.5rem;
            }
            .card {
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .card-header {
                background-color: #f8f9fa;
                border-bottom: 1px solid #e9ecef;
                font-weight: 600;
            }
            .btn-primary {
                background-color: #0d6efd;
                border-color: #0d6efd;
            }
            .btn-primary:hover {
                background-color: #0b5ed7;
                border-color: #0a58ca;
            }
            .tab-content {
                padding: 20px;
            }
            #widgetCode {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border: 1px solid #dee2e6;
            }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-light bg-light mb-4">
            <div class="container">
                <span class="navbar-brand">
                    <i class="fas fa-robot me-2"></i> Chatbot de Documentos con IA
                </span>
            </div>
        </nav>
        
        <div class="container">
            <ul class="nav nav-tabs" id="myTab" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="home-tab" data-bs-toggle="tab" data-bs-target="#home" type="button" role="tab" aria-controls="home" aria-selected="true">Chat</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="widget-tab" data-bs-toggle="tab" data-bs-target="#widget" type="button" role="tab" aria-controls="widget" aria-selected="false">Widget</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="settings-tab" data-bs-toggle="tab" data-bs-target="#settings" type="button" role="tab" aria-controls="settings" aria-selected="false">Configuración</button>
                </li>
            </ul>
            
            <div class="tab-content" id="myTabContent">
                <!-- Pestaña de Chat -->
                <div class="tab-pane fade show active" id="home" role="tabpanel" aria-labelledby="home-tab">
                    <div class="row">
                        <div class="col-md-8 offset-md-2">
                            <div id="uploadSection">
                                <div class="card mb-4">
                                    <div class="card-header">
                                        <i class="fas fa-file-upload me-2"></i> Sube un documento
                                    </div>
                                    <div class="card-body">
                                        <form id="uploadForm">
                                            <div class="mb-3">
                                                <label for="document" class="form-label">Selecciona un archivo (PDF, DOCX, TXT)</label>
                                                <input type="file" class="form-control" id="document" required>
                                            </div>
                                            <div class="mb-3">
                                                <label for="apiKeyInput" class="form-label">API Key de Deepseek (opcional)</label>
                                                <input type="password" class="form-control" id="apiKeyInput" placeholder="Ingresa tu API Key de Deepseek">
                                                <div class="form-text">Deja en blanco para usar la API Key configurada en el servidor.</div>
                                            </div>
                                            <button type="submit" class="btn btn-primary">
                                                <i class="fas fa-upload me-2"></i> Subir documento
                                            </button>
                                        </form>
                                    </div>
                                </div>
                            </div>
                            
                            <div id="chatSection" style="display: none;">
                                <div class="card mb-4">
                                    <div class="card-header d-flex justify-content-between align-items-center">
                                        <span><i class="fas fa-comments me-2"></i> Chat con tu documento</span>
                                        <span id="documentName" class="badge bg-info text-white"></span>
                                    </div>
                                    <div class="card-body">
                                        <div id="chatContainer" class="chat-container">
                                            <div class="bot-message">
                                                Hola, puedes hacerme preguntas sobre el documento que has subido. Estoy impulsado por IA para darte respuestas precisas.
                                            </div>
                                        </div>
                                        <form id="questionForm">
                                            <div class="input-group">
                                                <input type="text" id="question" class="form-control" placeholder="Escribe tu pregunta aquí..." required>
                                                <button class="btn btn-primary" type="submit">
                                                    <i class="fas fa-paper-plane"></i>
                                                </button>
                                            </div>
                                        </form>
                                    </div>
                                </div>
                                <button id="uploadNew" class="btn btn-outline-secondary">
                                    <i class="fas fa-file-upload me-2"></i> Subir otro documento
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Pestaña de Widget -->
                <div class="tab-pane fade" id="widget" role="tabpanel" aria-labelledby="widget-tab">
                    <div class="row">
                        <div class="col-md-8 offset-md-2">
                            <div class="card">
                                <div class="card-header">
                                    <i class="fas fa-code me-2"></i> Código del Widget
                                </div>
                                <div class="card-body">
                                    <p>Copia este código y pégalo en tu sitio web para integrar el chatbot:</p>
                                    <pre id="widgetCode" class="mb-3">
&lt;script src="/widget.js"&gt;&lt;/script&gt;
&lt;script&gt;
    document.addEventListener('DOMContentLoaded', function() {
        initChatbotWidget({
            apiKey: 'TU_API_KEY',
            theme: 'light', // light o dark
            position: 'bottom-right' // bottom-right, bottom-left, top-right, top-left
        });
    });
&lt;/script&gt;</pre>
                                    <p>Personaliza la apariencia del widget:</p>
                                    <div class="mb-3">
                                        <label class="form-label">Tema</label>
                                        <div class="form-check">
                                            <input class="form-check-input" type="radio" name="widgetTheme" id="lightTheme" value="light" checked>
                                            <label class="form-check-label" for="lightTheme">
                                                Claro
                                            </label>
                                        </div>
                                        <div class="form-check">
                                            <input class="form-check-input" type="radio" name="widgetTheme" id="darkTheme" value="dark">
                                            <label class="form-check-label" for="darkTheme">
                                                Oscuro
                                            </label>
                                        </div>
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label">Posición</label>
                                        <select class="form-select" id="widgetPosition">
                                            <option value="bottom-right" selected>Abajo a la derecha</option>
                                            <option value="bottom-left">Abajo a la izquierda</option>
                                            <option value="top-right">Arriba a la derecha</option>
                                            <option value="top-left">Arriba a la izquierda</option>
                                        </select>
                                    </div>
                                    <div class="mb-3">
                                        <label for="widgetApiKey" class="form-label">API Key</label>
                                        <input type="text" class="form-control" id="widgetApiKey" placeholder="Ingresa tu API Key">
                                    </div>
                                    <button id="updateWidgetCode" class="btn btn-primary">
                                        <i class="fas fa-sync-alt me-2"></i> Actualizar código
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Pestaña de Configuración -->
                <div class="tab-pane fade" id="settings" role="tabpanel" aria-labelledby="settings-tab">
                    <div class="row">
                        <div class="col-md-8 offset-md-2">
                            <div class="card">
                                <div class="card-header">
                                    <i class="fas fa-cog me-2"></i> Configuración
                                </div>
                                <div class="card-body">
                                    <form id="settingsForm">
                                        <div class="mb-3">
                                            <label for="deepseekApiKey" class="form-label">API Key de Deepseek</label>
                                            <input type="password" class="form-control" id="deepseekApiKey" placeholder="Ingresa tu API Key de Deepseek">
                                            <div class="form-text">Esta API Key se usará cuando no se proporcione una en el widget.</div>
                                        </div>
                                        <button type="submit" class="btn btn-primary">
                                            <i class="fas fa-save me-2"></i> Guardar configuración
                                        </button>
                                    </form>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            let currentDocumentId = null;
            let userApiKey = null;
            
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const fileInput = document.getElementById('document');
                const apiKeyInput = document.getElementById('apiKeyInput');
                
                if (fileInput.files.length === 0) {
                    alert('Por favor selecciona un documento');
                    return;
                }
                
                // Guardar API key si se proporciona
                if (apiKeyInput.value.trim()) {
                    userApiKey = apiKeyInput.value.trim();
                }
                
                const formData = new FormData();
                formData.append('document', fileInput.files[0]);
                
                try {
                    const response = await fetch('/upload-document/', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        currentDocumentId = result.document_id;
                        document.getElementById('documentName').textContent = fileInput.files[0].name;
                        document.getElementById('uploadSection').style.display = 'none';
                        document.getElementById('chatSection').style.display = 'block';
                    } else {
                        alert('Error: ' + result.detail);
                    }
                } catch (error) {
                    console.error('Error:', error);
                    alert('Ocurrió un error al subir el documento');
                }
            });
            
            document.getElementById('questionForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const questionInput = document.getElementById('question');
                const question = questionInput.value.trim();
                
                if (!question) return;
                
                // Añadir mensaje del usuario al chat
                const chatContainer = document.getElementById('chatContainer');
                const userMessageDiv = document.createElement('div');
                userMessageDiv.className = 'user-message';
                userMessageDiv.textContent = question;
                chatContainer.appendChild(userMessageDiv);
                
                // Añadir mensaje de "pensando..." del bot
                const thinkingDiv = document.createElement('div');
                thinkingDiv.className = 'bot-message thinking';
                thinkingDiv.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i> Procesando tu pregunta...';
                chatContainer.appendChild(thinkingDiv);
                
                // Scroll al final del chat
                chatContainer.scrollTop = chatContainer.scrollHeight;
                
                // Limpiar el input
                questionInput.value = '';
                
                try {
                    const requestBody = {
                        question: question,
                        document_id: currentDocumentId
                    };
                    
                    // Añadir API key si está disponible
                    if (userApiKey) {
                        requestBody.api_key = userApiKey;
                    }
                    
                    const response = await fetch('/ask-question/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(requestBody)
                    });
                    
                    const result = await response.json();
                    
                    // Eliminar el mensaje de "pensando..."
                    chatContainer.removeChild(thinkingDiv);
                    
                    // Añadir respuesta del bot al chat
                    const botMessageDiv = document.createElement('div');
                    botMessageDiv.className = 'bot-message';
                    
                    if (response.ok) {
                        botMessageDiv.textContent = result.answer;
                    } else {
                        botMessageDiv.textContent = 'Lo siento, no pude procesar tu pregunta. ' + result.detail;
                    }
                    
                    chatContainer.appendChild(botMessageDiv);
                    
                    // Scroll al final del chat
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                    
                } catch (error) {
                    console.error('Error:', error);
                    
                    // Eliminar el mensaje de "pensando..."
                    chatContainer.removeChild(thinkingDiv);
                    
                    // Mensaje de error en el chat
                    const errorMessageDiv = document.createElement('div');
                    errorMessageDiv.className = 'bot-message';
                    errorMessageDiv.textContent = 'Lo siento, ocurrió un error al procesar tu pregunta.';
                    chatContainer.appendChild(errorMessageDiv);
                    
                    // Scroll al final del chat
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            });
            
            document.getElementById('uploadNew').addEventListener('click', () => {
                document.getElementById('uploadSection').style.display = 'block';
                document.getElementById('chatSection').style.display = 'none';
                document.getElementById('document').value = '';
                document.getElementById('apiKeyInput').value = '';
                currentDocumentId = null;
            });
            
            // Actualizar código del widget
            document.getElementById('updateWidgetCode').addEventListener('click', () => {
                const theme = document.querySelector('input[name="widgetTheme"]:checked').value;
                const position = document.getElementById('widgetPosition').value;
                const apiKey = document.getElementById('widgetApiKey').value || 'TU_API_KEY';
                
                const code = `<script src="/widget.js"><\/script>
<script>
    document.addEventListener('DOMContentLoaded', function() {
        initChatbotWidget({
            apiKey: '${apiKey}',
            theme: '${theme}',
            position: '${position}'
        });
    });
<\/script>`;
                
                document.getElementById('widgetCode').textContent = code;
            });
            
            // Guardar configuración
            document.getElementById('settingsForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const apiKey = document.getElementById('deepseekApiKey').value.trim();
                
                try {
                    const response = await fetch('/api/settings/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            deepseek_api_key: apiKey
                        })
                    });
                    
                    if (response.ok) {
                        alert('Configuración guardada correctamente');
                    } else {
                        const result = await response.json();
                        alert('Error: ' + result.detail);
                    }
                } catch (error) {
                    console.error('Error:', error);
                    alert('Ocurrió un error al guardar la configuración');
                }
            });
        </script>
    </body>
    </html>
    """

# Ruta para subir documentos
@app.post("/upload-document/")
async def upload_document(document: UploadFile = File(...)):
    # Generar ID único para el documento
    document_id = str(uuid.uuid4())
    
    # Crear directorio para guardar el archivo
    file_path = f"uploads/{document_id}_{document.filename}"
    
    try:
        # Guardar el archivo
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(document.file, buffer)
        
        # Extraer texto del documento
        try:
            document_text = extract_text(file_path)
            
            # Procesar documento y generar embeddings
            chunks, chunk_embeddings = process_document(document_text)
            
            # Almacenar el texto y embeddings
            documents[document_id] = {
                "filename": document.filename,
                "path": file_path,
                "text": document_text,
                "chunks": chunks
            }
            
            document_embeddings[document_id] = chunk_embeddings
            
            return {"document_id": document_id, "message": "Documento subido correctamente"}
        
        except Exception as e:
            os.remove(file_path)  # Eliminar archivo si hay error
            raise HTTPException(statusos.remove(file_path)  # Eliminar archivo si hay error
            raise HTTPException(status_code=400, detail=f"Error al procesar el documento: {str(e)}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al subir el documento: {str(e)}")

# Ruta para hacer preguntas usando IA
@app.post("/ask-question/")
async def ask_question(question_data: Question):
    document_id = question_data.document_id
    question = question_data.question
    api_key = getattr(question_data, 'api_key', None)
    
    if document_id not in documents:
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    
    try:
        # Obtener los chunks y embeddings del documento
        chunks = documents[document_id]["chunks"]
        chunk_embeddings = document_embeddings[document_id]
        
        # Encontrar chunks relevantes para la pregunta
        relevant_chunks, relevance_scores = find_relevant_chunks(chunks, chunk_embeddings, question)
        
        # Si se proporciona una API key personalizada, usarla temporalmente
        global DEEPSEEK_API_KEY
        original_api_key = DEEPSEEK_API_KEY
        
        if api_key:
            DEEPSEEK_API_KEY = api_key
            
        try:
            # Consultar a Deepseek API con los chunks relevantes
            answer = query_deepseek(question, relevant_chunks)
        finally:
            # Restaurar la API key original
            DEEPSEEK_API_KEY = original_api_key
        
        return {"answer": answer}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la pregunta: {str(e)}")

# Ruta para guardar configuración
@app.post("/api/settings/")
async def save_settings(settings: dict):
    try:
        # En una aplicación real, esto se guardaría en una base de datos
        global DEEPSEEK_API_KEY
        if "deepseek_api_key" in settings and settings["deepseek_api_key"]:
            DEEPSEEK_API_KEY = settings["deepseek_api_key"]
            
        return {"status": "success", "message": "Configuración guardada correctamente"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar la configuración: {str(e)}")

# Ruta para servir el widget JavaScript
@app.get("/widget.js", response_class=Response(media_type="application/javascript"))
async def get_widget_js():
    return """
    (function() {
        // Crear estilos para el widget
        const style = document.createElement('style');
        style.textContent = `
            .chatbot-widget-container {
                position: fixed;
                z-index: 9999;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .chatbot-widget-container.bottom-right {
                bottom: 20px;
                right: 20px;
            }
            .chatbot-widget-container.bottom-left {
                bottom: 20px;
                left: 20px;
            }
            .chatbot-widget-container.top-right {
                top: 20px;
                right: 20px;
            }
            .chatbot-widget-container.top-left {
                top: 20px;
                left: 20px;
            }
            .chatbot-widget-button {
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background-color: #0d6efd;
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                transition: all 0.3s ease;
            }
            .chatbot-widget-button:hover {
                transform: scale(1.05);
            }
            .chatbot-widget-button i {
                font-size: 24px;
            }
            .chatbot-widget-panel {
                position: absolute;
                bottom: 70px;
                right: 0;
                width: 350px;
                height: 500px;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                display: none;
                flex-direction: column;
                overflow: hidden;
            }
            .chatbot-widget-header {
                padding: 15px;
                background-color: #f8f9fa;
                border-bottom: 1px solid #e9ecef;
                font-weight: bold;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .chatbot-widget-close {
                cursor: pointer;
                font-size: 20px;
            }
            .chatbot-widget-body {
                flex: 1;
                padding: 15px;
                overflow-y: auto;
            }
            .chatbot-widget-footer {
                padding: 10px 15px;
                border-top: 1px solid #e9ecef;
            }
            .chatbot-widget-input-group {
                display: flex;
            }
            .chatbot-widget-input {
                flex: 1;
                padding: 8px 12px;
                border: 1px solid #ced4da;
                border-radius: 4px 0 0 4px;
                outline: none;
            }
            .chatbot-widget-send {
                background-color: #0d6efd;
                color: white;
                border: none;
                border-radius: 0 4px 4px 0;
                padding: 8px 15px;
                cursor: pointer;
            }
            .chatbot-widget-message {
                margin-bottom: 15px;
                max-width: 80%;
                padding: 10px 15px;
                border-radius: 15px;
            }
            .chatbot-widget-user-message {
                background-color: #e3f2fd;
                margin-left: auto;
                border-radius: 15px 15px 0 15px;
                text-align: right;
            }
            .chatbot-widget-bot-message {
                background-color: #f1f1f1;
                border-radius: 15px 15px 15px 0;
            }
            .chatbot-widget-upload-section {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100%;
                padding: 20px;
                text-align: center;
            }
            .chatbot-widget-upload-label {
                margin-bottom: 15px;
                font-weight: 500;
            }
            .chatbot-widget-upload-btn {
                background-color: #0d6efd;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                cursor: pointer;
                margin-top: 10px;
            }
            .chatbot-widget-dark {
                color: #f8f9fa;
            }
            .chatbot-widget-dark .chatbot-widget-panel {
                background-color: #343a40;
            }
            .chatbot-widget-dark .chatbot-widget-header {
                background-color: #212529;
                border-bottom: 1px solid #495057;
            }
            .chatbot-widget-dark .chatbot-widget-footer {
                border-top: 1px solid #495057;
            }
            .chatbot-widget-dark .chatbot-widget-input {
                background-color: #495057;
                border: 1px solid #6c757d;
                color: #f8f9fa;
            }
            .chatbot-widget-dark .chatbot-widget-user-message {
                background-color: #0d6efd;
                color: white;
            }
            .chatbot-widget-dark .chatbot-widget-bot-message {
                background-color: #495057;
            }
        `;
        document.head.appendChild(style);
        
        // Añadir Font Awesome para iconos
        if (!document.querySelector('link[href*="font-awesome"]')) {
            const fontAwesome = document.createElement('link');
            fontAwesome.rel = 'stylesheet';
            fontAwesome.href = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css';
            document.head.appendChild(fontAwesome);
        }
        
        // Función para inicializar el widget
        window.initChatbotWidget = function(config) {
            const apiKey = config.apiKey || '';
            const theme = config.theme || 'light';
            const position = config.position || 'bottom-right';
            
            let currentDocumentId = null;
            let chatHistory = [];
            
            // Crear contenedor del widget
            const container = document.createElement('div');
            container.className = `chatbot-widget-container ${position} ${theme === 'dark' ? 'chatbot-widget-dark' : ''}`;
            
            // Crear botón del widget
            const button = document.createElement('div');
            button.className = 'chatbot-widget-button';
            button.innerHTML = '<i class="fas fa-comments"></i>';
            
            // Crear panel del chat
            const panel = document.createElement('div');
            panel.className = 'chatbot-widget-panel';
            
            // Crear encabezado del panel
            const header = document.createElement('div');
            header.className = 'chatbot-widget-header';
            header.innerHTML = '<span>Chatbot de Documentos</span><span class="chatbot-widget-close">&times;</span>';
            
            // Crear cuerpo del panel
            const body = document.createElement('div');
            body.className = 'chatbot-widget-body';
            
            // Crear sección de subida de documento
            const uploadSection = document.createElement('div');
            uploadSection.className = 'chatbot-widget-upload-section';
            uploadSection.innerHTML = `
                <div class="chatbot-widget-upload-label">Sube un documento para comenzar</div>
                <input type="file" id="chatbot-widget-file" style="display: none;">
                <button class="chatbot-widget-upload-btn">
                    <i class="fas fa-file-upload"></i> Seleccionar documento
                </button>
            `;
            
            // Crear pie del panel
            const footer = document.createElement('div');
            footer.className = 'chatbot-widget-footer';
            footer.innerHTML = `
                <div class="chatbot-widget-input-group">
                    <input type="text" class="chatbot-widget-input" placeholder="Escribe tu pregunta...">
                    <button class="chatbot-widget-send"><i class="fas fa-paper-plane"></i></button>
                </div>
            `;
            
            // Añadir elementos al DOM
            panel.appendChild(header);
            panel.appendChild(body);
            body.appendChild(uploadSection);
            panel.appendChild(footer);
            container.appendChild(button);
            container.appendChild(panel);
            document.body.appendChild(container);
            
            // Referencia a elementos interactivos
            const closeBtn = header.querySelector('.chatbot-widget-close');
            const fileInput = uploadSection.querySelector('#chatbot-widget-file');
            const uploadBtn = uploadSection.querySelector('.chatbot-widget-upload-btn');
            const questionInput = footer.querySelector('.chatbot-widget-input');
            const sendBtn = footer.querySelector('.chatbot-widget-send');
            
            // Manejadores de eventos
            button.addEventListener('click', () => {
                panel.style.display = 'flex';
            });
            
            closeBtn.addEventListener('click', () => {
                panel.style.display = 'none';
            });
            
            uploadBtn.addEventListener('click', () => {
                fileInput.click();
            });
            
            fileInput.addEventListener('change', async (e) => {
                if (e.target.files.length === 0) return;
                
                const file = e.target.files[0];
                const formData = new FormData();
                formData.append('document', file);
                
                try {
                    const response = await fetch('/upload-document/', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        currentDocumentId = result.document_id;
                        
                        // Limpiar sección de subida y mostrar chat
                        uploadSection.style.display = 'none';
                        
                        // Añadir mensaje de bienvenida
                        addBotMessage('Hola, puedes hacerme preguntas sobre el documento que has subido.');
                        
                        // Habilitar envío de preguntas
                        footer.style.display = 'block';
                    } else {
                        alert('Error: ' + result.detail);
                    }
                } catch (error) {
                    console.error('Error:', error);
                    alert('Ocurrió un error al subir el documento');
                }
            });
            
            sendBtn.addEventListener('click', sendQuestion);
            questionInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    sendQuestion();
                }
            });
            
            // Inicialización
            footer.style.display = 'none';
            
            // Función para enviar pregunta
            async function sendQuestion() {
                const question = questionInput.value.trim();
                if (!question || !currentDocumentId) return;
                
                // Añadir mensaje del usuario
                addUserMessage(question);
                
                // Limpiar input
                questionInput.value = '';
                
                // Añadir mensaje de "pensando..."
                const thinkingId = addBotMessage('<i class="fas fa-spinner fa-spin"></i> Procesando tu pregunta...', true);
                
                try {
                    const response = await fetch('/ask-question/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            question: question,
                            document_id: currentDocumentId,
                            api_key: apiKey
                        })
                    });
                    
                    const result = await response.json();
                    
                    // Eliminar mensaje de "pensando..."
                    removeBotMessage(thinkingId);
                    
                    if (response.ok) {
                        addBotMessage(result.answer);
                    } else {
                        addBotMessage('Lo siento, no pude procesar tu pregunta. ' + result.detail);
                    }
                } catch (error) {
                    console.error('Error:', error);
                    
                    // Eliminar mensaje de "pensando..."
                    removeBotMessage(thinkingId);
                    
                    addBotMessage('Lo siento, ocurrió un error al procesar tu pregunta.');
                }
            }
            
            // Función para añadir mensaje del usuario
            function addUserMessage(text) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'chatbot-widget-message chatbot-widget-user-message';
                messageDiv.textContent = text;
                body.appendChild(messageDiv);
                
                // Scroll al final
                body.scrollTop = body.scrollHeight;
                
                // Guardar en historial
                chatHistory.push({role: 'user', content: text});
            }
            
            // Función para añadir mensaje del bot
            function addBotMessage(html, isThinking = false) {
                const messageId = 'msg-' + Date.now();
                const messageDiv = document.createElement('div');
                messageDiv.className = 'chatbot-widget-message chatbot-widget-bot-message';
                if (isThinking) messageDiv.classList.add('thinking');
                messageDiv.innerHTML = html;
                messageDiv.id = messageId;
                body.appendChild(messageDiv);
                
                // Scroll al final
                body.scrollTop = body.scrollHeight;
                
                if (!isThinking) {
                    // Guardar en historial
                    chatHistory.push({role: 'assistant', content: html});
                }
                
                return messageId;
            }
            
            // Función para eliminar mensaje del bot
            function removeBotMessage(messageId) {
                const messageDiv = document.getElementById(messageId);
                if (messageDiv) {
                    body.removeChild(messageDiv);
                }
            }
        };
    })();
    """

# Ruta para una página de demostración del widget
@app.get("/widget-demo", response_class=HTMLResponse)
async def get_widget_demo():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Demo del Widget de Chatbot</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container my-5">
            <h1 class="text-center mb-4">Demostración del Widget de Chatbot</h1>
            <div class="row">
                <div class="col-md-8 offset-md-2">
                    <div class="card">
                        <div class="card-body">
                            <p>Esta es una página de demostración del widget de chatbot. Haz clic en el ícono de chat en la esquina inferior derecha para interactuar con el chatbot.</p>
                            <p>Puedes subir un documento y hacer preguntas sobre él.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script src="/widget.js"></script>
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                initChatbotWidget({
                    apiKey: '',  // Usar la API key configurada en el servidor
                    theme: 'light',
                    position: 'bottom-right'
                });
            });
        </script>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Punto de entrada para ejecutar la aplicación
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
