# ==============================================================================
# SCRIPT COMPLETO E OTIMIZADO PARA O ASSISTENTE DE PESQUISA
# ==============================================================================

import streamlit as st
import os
import shutil
from dotenv import load_dotenv

# Importações atualizadas do Langchain para criar um Agente
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- CONFIGURAÇÕES E CONSTANTES ---

# Caminho onde o índice FAISS vetorial será salvo e carregado
FAISS_INDEX_PATH = "faiss_index"

# --- FUNÇÕES DE LÓGICA DO AGENTE OTIMIZADAS ---

def configure_llm_and_embeddings(api_key):
    """Inicializa o modelo LLM (GPT) e os Embeddings com a chave de API da OpenAI."""
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, max_tokens=1500, temperature=0.7)
        embeddings = OpenAIEmbeddings(api_key=api_key)
        return llm, embeddings
    except Exception as e:
        st.error(f"Erro ao inicializar os modelos da OpenAI: {e}")
        return None, None

def create_and_save_vector_store(embeddings_model):
    """
    Lê os arquivos da pasta 'docs', cria a base de conhecimento vetorial (Vector Store)
    e a salva em disco para uso futuro.
    """
    docs_path = "docs"
    if not os.path.exists(docs_path):
        st.sidebar.error(f"A pasta '{docs_path}' não foi encontrada.")
        return None

    documents, failed_files = [], []
    total_files_processed = 0
    
    spinner_text = 'Lendo e processando arquivos da pasta "docs" para criar uma nova base de conhecimento...'
    with st.spinner(spinner_text):
        for dirpath, _, filenames in os.walk(docs_path):
            for file_name in filenames:
                file_path = os.path.join(dirpath, file_name)
                try:
                    loader = None
                    if file_name.endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                    elif file_name.endswith('.docx'):
                        loader = UnstructuredWordDocumentLoader(file_path)
                    elif file_name.endswith('.pptx'):
                        loader = UnstructuredPowerPointLoader(file_path)
                    
                    if loader:
                        documents.extend(loader.load())
                        total_files_processed += 1
                except Exception as e:
                    failed_files.append((file_name, str(e)))
    
    for file_name, error_msg in failed_files:
        st.sidebar.error(f"Erro ao processar '{file_name}'. O arquivo foi ignorado.")

    if not documents:
        st.sidebar.warning("Nenhum documento foi encontrado ou pôde ser processado com sucesso.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    texts = text_splitter.split_documents(documents)
    
    # Cria o Vector Store a partir dos documentos
    vector_store = FAISS.from_documents(texts, embeddings_model)
    
    # Salva o Vector Store localmente no caminho definido
    vector_store.save_local(FAISS_INDEX_PATH)
    
    st.sidebar.success(f"Nova base de conhecimento criada e salva com {total_files_processed} arquivo(s).")
    return vector_store

def load_or_create_vector_store(embeddings_model):
    """
    Carrega o Vector Store do disco se ele existir.
    Caso contrário, cria um novo e o salva no disco.
    Utiliza o cache do Streamlit para evitar recarregamentos desnecessários durante a mesma sessão.
    """
    # Usamos o cache para esta função, garantindo que ela rode apenas uma vez por sessão.
    @st.cache_resource(show_spinner="Carregando base de conhecimento...")
    def _load_or_create_cached():
        if os.path.exists(FAISS_INDEX_PATH):
            st.sidebar.info("Carregando base de conhecimento existente do disco.")
            # O parâmetro allow_dangerous_deserialization é necessário para o FAISS
            return FAISS.load_local(FAISS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)
        else:
            st.sidebar.info("Nenhuma base de conhecimento encontrada. Criando uma nova.")
            return create_and_save_vector_store(embeddings_model)
            
    return _load_or_create_cached()

# --- INTERFACE GRÁFICA COM STREAMLIT ---

load_dotenv()
st.set_page_config(page_title="Assistente Criativo de Pesquisa", layout="wide")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] > .main { background-color: #f0f2f6; }
    [data-testid="stChatInput"] textarea { height: 100px; font-size: 16px; border-radius: 10px; }
    h1 { color: #1E3A8A; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Assistente Criativo de Pesquisa com GPT")
st.write("Pergunte-me qualquer coisa! Eu pesquisarei nos seus documentos e na internet para fornecer respostas completas e criativas.")

with st.sidebar:
    st.header("Base de Conhecimento")
    st.info(f"A base de dados é carregada da pasta local `{FAISS_INDEX_PATH}`.")
    
    if st.button("Reconstruir Base de Conhecimento"):
        if os.path.exists(FAISS_INDEX_PATH):
            with st.spinner("Removendo base de conhecimento antiga..."):
                shutil.rmtree(FAISS_INDEX_PATH)
            st.cache_resource.clear()  # Limpa todo o cache de recursos do Streamlit
            st.success("Base de conhecimento antiga removida! A aplicação será reiniciada para criar uma nova.")
            st.rerun()  # Força a reinicialização do script
        else:
            st.warning("Nenhuma base de conhecimento para remover. A aplicação será reiniciada para criar uma.")
            st.cache_resource.clear()
            st.rerun()

    st.header("Conexão com a Internet")
    st.success("Este agente usa a API Tavily para buscar informações atualizadas na web.")

# --- LÓGICA PRINCIPAL DO AGENTE ---

openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not openai_api_key or not tavily_api_key:
    st.error("Chaves de API não encontradas!")
    st.info("Por favor, configure as variáveis `OPENAI_API_KEY` e `TAVILY_API_KEY` no seu arquivo .env.")
    st.stop()

llm, embeddings = configure_llm_and_embeddings(openai_api_key)
if llm is None or embeddings is None:
    st.stop()

# MODIFICADO: Chamamos a nova função que gerencia o carregamento ou criação da base
vector_store = load_or_create_vector_store(embeddings)

if vector_store is None:
    st.warning("A base de conhecimento não pôde ser carregada ou criada. A busca em documentos está desativada.")
    tools = [TavilySearchResults(max_results=3, tavily_api_key=tavily_api_key)]
else:
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    retriever_tool = create_retriever_tool(
        retriever,
        "document_search",
        "Busca informações na base de conhecimento local. Use esta ferramenta para qualquer pergunta sobre os arquivos fornecidos."
    )
    # A lista de ferramentas agora depende da existência do vector_store
    tools = [retriever_tool, TavilySearchResults(max_results=3, tavily_api_key=tavily_api_key)]

# Cria o prompt do agente
prompt = ChatPromptTemplate.from_messages([
    ("system", """Você é um assistente de pesquisa IA avançado e criativo.
Sua missão é fornecer respostas longas, detalhadas e bem estruturadas.
Estratégia:
1.  **Priorize a ferramenta `document_search`** para responder com base nos arquivos fornecidos.
2.  Se a informação não estiver nos documentos, use a `tavily_search_results_json` para pesquisar na internet.
3.  Sintetize as informações de ambas as fontes, se necessário, para criar a resposta mais completa possível.
4.  Seja criativo e analítico. Não se limite a repetir a informação; explique, conecte ideias e ofereça insights.
5.  Sempre cite suas fontes, seja o nome do arquivo local ou o link da web."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Lógica do Chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Estou pronto para pesquisar e criar. Qual é a sua pergunta?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Digite sua pergunta aqui..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pesquisando e elaborando a resposta..."):
            response = agent_executor.invoke({"input": user_prompt})
            response_text = response['output']
            st.markdown(response_text)
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})