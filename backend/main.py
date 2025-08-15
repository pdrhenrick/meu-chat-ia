# backend/main.py - VERSÃO FINAL
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)

try:
    if os.getenv("SERPAPI_API_KEY"):
        search = SerpAPIWrapper()
        print("INFO: Usando ferramenta de busca profissional: SerpAPI")
    else:
        search = DuckDuckGoSearchRun()
        print("INFO: Usando ferramenta de busca gratuita: DuckDuckGo")
except Exception as e:
    print(f"AVISO: Erro ao inicializar ferramenta de busca. O bot não funcionará. Erro: {e}")
    search = None

template = """Você é um assistente de IA especialista em resumir resultados de busca para responder perguntas. Use os trechos de contexto da busca abaixo para responder a pergunta do usuário de forma clara e completa. Responda em português brasileiro.
Contexto da Busca:{context}
Pergunta do Usuário:{question}
Sua Resposta:"""
prompt = ChatPromptTemplate.from_template(template)

chain = (RunnableParallel(context=(lambda x: x["question"]) | search, question=RunnablePassthrough()) | prompt | llm | StrOutputParser())
app = FastAPI(title="API do Agente de Pesquisa")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel): question: str

@app.post("/chat")
async def chat_with_search_agent(request: ChatRequest):
    if search is None: return {"answer": "Desculpe, a ferramenta de busca não está configurada."}
    try:
        response = chain.invoke({"question": request.question})
        return {"answer": response}
    except Exception as e: return {"answer": f"Ocorreu um erro na busca: {e}"}