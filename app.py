# =======================================================================================
# SCRIPT COMPLETO E OTIMIZADO PARA O ASSISTENTE DE PESQUISA (COM MEMÓRIA E GPT-4)
# Implementações e Refatoração por Prof. Inteligência Artificial
# VERSÃO COM BUSCA PARALELA MULTI-QUERY E DOWNLOAD DE ARQUIVOS
# =======================================================================================

import streamlit as st
import os
import shutil
import re
from dotenv import load_dotenv
from collections import defaultdict

# Importações do Langchain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationSummaryBufferMemory
import logging

# Configuração do logging para depuração (útil para ver as sub-perguntas geradas)
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


# --- CONFIGURAÇÕES E CONSTANTES ---

FAISS_INDEX_PATH = "faiss_index"
DOCS_FOLDER = "docs"

# --- FUNÇÕES DE LÓGICA DO AGENTE OTIMIZADAS ---

def configure_llm_and_embeddings(api_key):
    """Inicializa o modelo LLM (GPT) e os Embeddings."""
    try:
        llm = ChatOpenAI(model="gpt-4", api_key=api_key, max_tokens=2000, temperature=0.5)
        embeddings = OpenAIEmbeddings(api_key=api_key)
        return llm, embeddings
    except Exception as e:
        st.error(f"Erro ao inicializar os modelos da OpenAI: {e}")
        return None, None

def create_and_save_vector_store(embeddings_model):
    """Lê, "documenta" (extraindo metadados) e cria o Vector Store."""
    if not os.path.exists(DOCS_FOLDER) or not os.listdir(DOCS_FOLDER):
        st.sidebar.error(f"A pasta '{DOCS_FOLDER}' está vazia. Adicione documentos.")
        return None

    all_docs = []
    failed_files = []
    
    with st.spinner('Processando e "documentando" arquivos para a base de conhecimento...'):
        for dirpath, _, filenames in os.walk(DOCS_FOLDER):
            for file_name in filenames:
                file_path = os.path.join(dirpath, file_name)
                try:
                    loader = None
                    if file_name.endswith(".pdf"):
                        loader = PyPDFLoader(file_path)
                    elif file_name.endswith(".docx"):
                        loader = UnstructuredWordDocumentLoader(file_path)
                    elif file_name.endswith(".pptx"):
                        loader = UnstructuredPowerPointLoader(file_path)
                    elif file_name.endswith(".xlsx"):
                        loader = UnstructuredExcelLoader(file_path, mode="elements")
                    
                    if loader:
                        docs = loader.load()
                        # "Documentação": Adiciona metadados cruciais a cada pedaço do documento
                        for doc in docs:
                            doc.metadata['source'] = file_path
                            doc.metadata['file_name'] = file_name
                            doc.metadata['folder'] = os.path.basename(dirpath)
                        all_docs.extend(docs)
                except Exception as e:
                    failed_files.append(f"{file_name} ({e})")

    if not all_docs:
        st.sidebar.warning("Nenhum documento foi processado com sucesso.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    texts = text_splitter.split_documents(all_docs)
    
    vector_store = FAISS.from_documents(texts, embeddings_model)
    vector_store.save_local(FAISS_INDEX_PATH)
    
    st.sidebar.success(f"Base de conhecimento criada com {len(set(doc.metadata['source'] for doc in all_docs))} arquivo(s).")
    return vector_store

@st.cache_resource(show_spinner="Carregando base de conhecimento...")
def load_or_create_vector_store(_embeddings_model):
    """Carrega ou cria o Vector Store, usando o cache do Streamlit."""
    if os.path.exists(FAISS_INDEX_PATH):
        st.sidebar.info("Carregando base de conhecimento existente.")
        return FAISS.load_local(FAISS_INDEX_PATH, _embeddings_model, allow_dangerous_deserialization=True)
    else:
        st.sidebar.info("Criando uma nova base de conhecimento.")
        return create_and_save_vector_store(_embeddings_model)

# --- INTERFACE GRÁFICA COM STREAMLIT ---

load_dotenv()
st.set_page_config(page_title="Assistente de Pesquisa Inteligente", layout="wide")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] > .main { background-color: #f0f2f6; }
    [data-testid="stChatInput"] textarea { height: 100px; font-size: 16px; border-radius: 10px; }
    h1 { color: #1E3A8A; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Assistente de Pesquisa com Busca Paralela")
st.write("Faça uma pergunta e eu quebrarei ela em múltiplas buscas paralelas para encontrar a melhor resposta em seus documentos.")

with st.sidebar:
    st.header("Base de Conhecimento")
    st.info(f"Os documentos são da pasta `{DOCS_FOLDER}`.")
    
    if st.button("Reconstruir Base de Conhecimento"):
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)
        st.cache_resource.clear()
        st.success("Base de conhecimento agendada para reconstrução! A aplicação será reiniciada.")
        st.rerun()

    st.header("Conexão com a Internet")
    st.success("Este agente usa a API Tavily para buscar informações na web se necessário.")

# --- LÓGICA PRINCIPAL DO AGENTE ---

openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not openai_api_key or not tavily_api_key:
    st.error("Chaves de API não configuradas no arquivo .env!")
    st.stop()

llm, embeddings = configure_llm_and_embeddings(openai_api_key)
if not llm or not embeddings:
    st.stop()

vector_store = load_or_create_vector_store(embeddings)
tools = []

if vector_store:
    # Prompt para o LLM gerar as sub-perguntas para a busca paralela
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""Você é um assistente de IA que gera múltiplas perguntas de busca a partir de uma única pergunta do usuário.
        Gere 3 versões diferentes da pergunta do usuário para buscar em uma base de dados vetorial.
        As perguntas devem ter perspectivas diferentes para maximizar a chance de encontrar documentos relevantes.
        Foque em extrair as palavras-chave e conceitos principais.
        Pergunta Original: {question}
        """,
    )
    
    # Cria o retriever MultiQuery que executa as buscas em paralelo
    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 7, "score_threshold": 0.35}
        ),
        llm=llm,
        prompt=QUERY_PROMPT
    )

    @tool
    def document_search_tool(query: str) -> str:
        """
        Busca nos documentos locais usando múltiplas queries paralelas e retorna o conteúdo agrupado por arquivo de origem.
        Use esta ferramenta como sua primeira e principal opção para responder a qualquer pergunta.
        """
        docs = retriever.invoke(query)
        if not docs:
            return "Nenhuma informação relevante foi encontrada nos documentos locais."
        
        grouped_results = defaultdict(list)
        for doc in docs:
            source = doc.metadata.get('source', 'Fonte desconhecida')
            grouped_results[source].append(doc.page_content)

        formatted_output = ""
        for source, contents in grouped_results.items():
            folder = os.path.basename(os.path.dirname(source))
            file_name = os.path.basename(source)
            formatted_output += f"INFORMAÇÃO ENCONTRADA:\n- Arquivo: '{file_name}'\n- Pasta: '{folder}'\n- Caminho Completo: '{source}'\n- Conteúdo Relevante:\n"
            for content in contents:
                formatted_output += f"  - \"{content}\"\n"
            formatted_output += "---\n"
        return formatted_output

    tools.append(document_search_tool)

tools.append(TavilySearchResults(max_results=3, api_key=tavily_api_key))

prompt = ChatPromptTemplate.from_messages([
    ("system", """Você é "Professor Inteligência Artificial", um especialista em IA e ciência de dados.

**DIRETIVA DE AÇÃO ESTRITA:**

**Passo 1: SAUDAÇÃO E CONTEXTO.**
- **Sempre** inicie sua resposta com uma saudação cordial (ex: "Olá! Com prazer...").
- **Demonstre Memória:** Se houver histórico de chat, após a saudação, conecte a pergunta atual ao tópico anterior.

**Passo 2: BUSCA INTERNA PARALELA OBRIGATÓRIA.**
- Sua primeira ação de pesquisa é **SEMPRE** usar a ferramenta `document_search_tool`.
- Analise a saída, que conterá o 'Caminho Completo:' do arquivo.

**Passo 3: FORMULAÇÃO DA RESPOSTA.**
- Baseie sua resposta nas informações da ferramenta.
- Formate a resposta de forma longa e detalhada usando markdown (títulos, listas, negrito).
- **CITAÇÃO E LINK DE DOWNLOAD:** Ao usar informação de um documento, cite a fonte com um link markdown usando o 'Caminho Completo:'.
  - **Formato:** `[Baixar Fonte: nome_do_arquivo.pdf](caminho_completo/do/arquivo.pdf)`

**Passo 4: BUSCA EXTERNA (PLANO B).**
- Apenas se a busca interna não retornar resultados, use a `tavily_search_results_json`.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

if "memory" not in st.session_state:
    st.session_state.memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=4000, 
        memory_key="chat_history", 
        return_messages=True
    )

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Sou o Professor Inteligência Artificial. Por favor, faça sua pergunta sobre os documentos."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_prompt := st.chat_input("Digite sua pergunta aqui..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Gerando sub-perguntas e buscando em paralelo..."):
            
            chat_history = st.session_state.memory.load_memory_variables({}).get("chat_history", [])
            
            try:
                response = agent_executor.invoke({
                    "input": user_prompt,
                    "chat_history": chat_history
                })
                response_text = response.get('output', 'Desculpe, não consegui processar sua solicitação.')
            except Exception as e:
                st.error(f"Ocorreu um erro inesperado: {e}")
                response_text = "Peço desculpas, mas encontrei um erro técnico."

            st.session_state.memory.save_context(
                {"input": user_prompt},
                {"output": response_text}
            )

            # Pós-processamento para criar botões de download
            file_path_pattern = r'\[Baixar Fonte:.*?\]\((.*?)\)'
            found_files = set(re.findall(file_path_pattern, response_text))

            st.markdown(response_text, unsafe_allow_html=True)

            if found_files:
                st.markdown("--- \n**Downloads dos Arquivos Citados:**")
                for file_path in found_files:
                    normalized_path = os.path.normpath(file_path)
                    if os.path.exists(normalized_path):
                        with open(normalized_path, "rb") as f:
                            file_bytes = f.read()
                        
                        file_name = os.path.basename(normalized_path)
                        st.download_button(
                            label=f"Download: '{file_name}'",
                            data=file_bytes,
                            file_name=file_name,
                            key=f"download_{file_name}_{user_prompt}" 
                        )
                    else:
                        st.warning(f"Arquivo citado '{normalized_path}' não encontrado para download.")

    st.session_state.messages.append({"role": "assistant", "content": response_text})