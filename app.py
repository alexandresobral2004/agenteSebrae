import streamlit as st
import os
from dotenv import load_dotenv

# Importações atualizadas do Langchain para criar um Agente
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_openai_tools_agent
# from langchain_tavily import TavilySearchResults
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- FUNÇÕES DE LÓGICA DO AGENTE ---

def configure_agent(api_key):
    """Inicializa o modelo LLM (GPT) e os Embeddings com a chave de API da OpenAI."""
    try:
        # Modelo atualizado para gpt-3.5-turbo e max_tokens aumentado para respostas mais longas
        llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, max_tokens=1500, temperature=0.7)
        embeddings = OpenAIEmbeddings(api_key=api_key)
        return llm, embeddings
    except Exception as e:
        st.error(f"Erro ao inicializar os modelos da OpenAI: {e}")
        return None, None

@st.cache_resource
def create_vector_store_from_docs(_embeddings_model):
    """
    Lê os arquivos da pasta 'docs' e de todas as suas subpastas de forma recursiva,
    e cria uma base de conhecimento vetorial (Vector Store).
    """
    docs_path = "docs"
    if not os.path.exists(docs_path):
        return None

    documents, failed_files = [], []
    total_files_processed = 0
    
    spinner_text = 'Lendo e processando arquivos da pasta "docs" e suas subpastas...'
    with st.spinner(spinner_text):
        # Usa os.walk() para percorrer o diretório e todas as suas subpastas
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
                    
                    # Se um carregador foi definido, processa o arquivo
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
    
    vector_store = FAISS.from_documents(texts, _embeddings_model)
    # Atualiza a mensagem de sucesso com a contagem correta de arquivos
    st.sidebar.success(f"Base de conhecimento criada com {total_files_processed} arquivo(s).")
    return vector_store

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
    st.info("Os arquivos da sua pasta `docs` e de todas as suas subpastas são a fonte primária de informação.")
    st.header("Conexão com a Internet")
    st.success("Este agente usa a API Tavily para buscar informações atualizadas na web.")

# --- LÓGICA PRINCIPAL DO AGENTE ---

openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not openai_api_key or not tavily_api_key:
    st.error("Chaves de API não encontradas!")
    st.info("Por favor, configure as variáveis `OPENAI_API_KEY` e `TAVILY_API_KEY` no seu ambiente.")
    st.stop()

llm, embeddings = configure_agent(openai_api_key)
if llm is None or embeddings is None:
    st.stop()

# Define as ferramentas que o agente pode usar
tools = [TavilySearchResults(max_results=3, tavily_api_key=tavily_api_key)]
vector_store = create_vector_store_from_docs(embeddings)

if vector_store:
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    retriever_tool = create_retriever_tool(
        retriever,
        "document_search",
        "Busca informações na base de conhecimento local. Use esta ferramenta para qualquer pergunta sobre os arquivos fornecidos."
    )
    tools.insert(0, retriever_tool) # Adiciona a ferramenta de documentos como prioridade

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

