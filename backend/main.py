# backend/main.py - VERSÃO FINAL CORRIGIDA

import os
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# NOVAS IMPORTAÇÕES PARA O AGENTE INTELIGENTE
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun

# Importações que ainda usamos
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate # Usaremos este para criar o prompt diretamente

load_dotenv()

# --- 1. INICIALIZAÇÃO DO CÉREBRO (LLM) ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)


# --- 2. DEFINIÇÃO DAS FERRAMENTAS DO AGENTE ---
# FERRAMENTA 0: Relógio e Calendário
def get_current_datetime(query: str = "") -> str:
    now = datetime.now()
    dias_semana = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"]
    meses = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]
    dia_semana_str = dias_semana[now.weekday()]
    mes_str = meses[now.month - 1]
    return f"A data de hoje é {dia_semana_str}, {now.day} de {mes_str} de {now.year}, e a hora atual é {now.strftime('%H:%M:%S')}."

datetime_tool = Tool(
    name="Relógio e Calendário",
    func=get_current_datetime,
    description="Essencial para obter a data e a hora atuais. Use esta ferramenta sempre que o usuário perguntar 'que dia é hoje?', 'que horas são?', 'qual a data atual?' ou qualquer variação similar."
)

# FERRAMENTA 1: Busca na Internet
internet_tool = None
try:
    if os.getenv("SERPAPI_API_KEY"):
        search = SerpAPIWrapper()
        print("INFO: Usando ferramenta de busca: SerpAPI")
    else:
        search = DuckDuckGoSearchRun()
        print("INFO: Usando ferramenta de busca gratuita: DuckDuckGo")

    internet_tool = Tool(
        name="Busca na Internet",
        func=search.run,
        description="Útil para encontrar informações sobre eventos recentes, notícias, fatos e qualquer tópico geral. Use para perguntas que a base de dados local provavelmente não conhece."
    )
except Exception as e:
    print(f"AVISO: Erro ao inicializar ferramenta de busca na internet: {e}")

# FERRAMENTA 2: Busca no Banco de Dados Local (dados.txt)
local_data_tool = None
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = TextLoader("../dados.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def retrieve_local_data(query: str) -> str:
        docs = retriever.invoke(query)
        if not docs:
            return "Nenhuma informação encontrada na base de dados local sobre isso."
        return "\n---\n".join([doc.page_content for doc in docs])

    local_data_tool = Tool(
        name="Busca na Base de Dados Local",
        func=retrieve_local_data,
        description="Útil para encontrar informações específicas sobre a Loja do Pedro, seus produtos como ZapFlex, horários de funcionamento e contatos. Use esta ferramenta para responder perguntas sobre esses tópicos."
    )
    print("INFO: Banco de dados vetorial local criado com sucesso.")
except Exception as e:
    print(f"AVISO: Erro ao criar o banco de dados vetorial local: {e}")


# --- 3. MONTAGEM FINAL DO AGENTE ---
tools = [datetime_tool]
if internet_tool:
    tools.append(internet_tool)
if local_data_tool:
    tools.append(local_data_tool)

# ESTA É A CORREÇÃO: O prompt está agora "embutido" no código
prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


# --- 4. API (ENDPOINT) ---
app = FastAPI(title="API do Agente Inteligente")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "API do Agente Inteligente está rodando! Vá para /docs para testar."}


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    try:
        response = agent_executor.invoke({"input": request.question})
        return {"answer": response.get("output", "Desculpe, não consegui processar a resposta.")}
    except Exception as e:
        return {"answer": f"Ocorreu um erro crítico no agente: {e}"}