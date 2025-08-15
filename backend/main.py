# backend/main.py - VERSÃO FINAL: AGENTE DE PESQUISA NA WEB

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Importações necessárias para a cadeia de pesquisa
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

# --- 1. CONFIGURAÇÃO DO CÉREBRO (LLM) E DA FERRAMENTA DE BUSCA ---

# Inicializa o modelo de IA, usando a versão "Flash" para máxima velocidade
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)

# Inicializa a ferramenta de busca (SerpAPI se tiver a chave, senão DuckDuckGo)
try:
    if os.getenv("SERPAPI_API_KEY"):
        search = SerpAPIWrapper()
        print("INFO: Usando ferramenta de busca: SerpAPI")
    else:
        search = DuckDuckGoSearchRun()
        print("INFO: Usando ferramenta de busca gratuita: DuckDuckGo")
except Exception as e:
    print(f"AVISO: Erro ao inicializar ferramenta de busca. O bot não funcionará. Erro: {e}")
    search = None

# --- 2. AS INSTRUÇÕES (PROMPT) ---
# Damos instruções claras para a IA resumir os resultados da busca
template = """
Você é um assistente de IA especialista em resumir resultados de busca para responder perguntas.
Use os trechos de contexto da busca abaixo para responder a pergunta do usuário de forma clara e concisa.
Responda em português brasileiro.

Contexto da Busca:
{context}

Pergunta do Usuário:
{question}

Sua Resposta:
"""
prompt = ChatPromptTemplate.from_template(template)

# --- 3. A LINHA DE MONTAGEM (CHAIN) ---
# Esta é a lógica principal:
# 1. A pergunta do usuário (`question`) é enviada para a ferramenta de busca (`search`) para obter o `context`.
# 2. A pergunta original também é mantida (`RunnablePassthrough`).
# 3. O `context` e a `question` são inseridos no `prompt`.
# 4. O `prompt` preenchido é enviado para a IA (`llm`).
# 5. A resposta da IA é convertida para texto (`StrOutputParser`).
chain = (
    RunnableParallel(
        context=(lambda x: x["question"]) | search,
        question=RunnablePassthrough(),
    )
    | prompt
    | llm
    | StrOutputParser()
)

# --- 4. A API (FastAPI) ---
app = FastAPI(title="API do Agente de Pesquisa")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_with_search_agent(request: ChatRequest):
    if search is None:
        return {"answer": "Desculpe, a ferramenta de busca não está configurada corretamente."}
    
    try:
        # Chamamos nossa nova 'chain' de pesquisa
        response = chain.invoke({"question": request.question})
        return {"answer": response}
    except Exception as e:
        return {"answer": f"Ocorreu um erro na busca: {e}"}