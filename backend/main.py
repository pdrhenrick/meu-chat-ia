# backend/main.py - VERSÃO FINAL: GÊNIO DE CONHECIMENTO GERAL

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Importações simplificadas
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Carrega as variáveis de ambiente (sua chave de API)
load_dotenv()

# --- 1. CONFIGURAÇÃO DO CÉREBRO (LLM) ---
# Inicializa o modelo de IA, usando a versão "Flash" para máxima velocidade
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)

# --- 2. AS INSTRUÇÕES (PROMPT) ---
# Damos instruções diretas e simples para a IA
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente de IA de conhecimento geral, similar ao Gemini e ao ChatGPT. Seu objetivo é responder às perguntas dos usuários da forma mais completa, precisa e prestativa possível sobre qualquer assunto."),
    ("user", "{input}")
])

# --- 3. A LINHA DE MONTAGEM (CHAIN) ---
# Criamos uma "linha direta" simples: Pergunta -> Instruções -> IA -> Resposta em texto
chain = prompt | llm | StrOutputParser()

# --- 4. A API (FastAPI) ---
app = FastAPI(title="API do Assistente de Conhecimento Geral")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_with_genius(request: ChatRequest):
    try:
        # Chamamos a nossa nova 'chain' simples
        response = chain.invoke({"input": request.question})
        return {"answer": response}
    except Exception as e:
        return {"answer": f"Ocorreu um erro na IA: {e}"}