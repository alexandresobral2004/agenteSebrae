Claro\! Aqui está o conteúdo completo do `README.md` em formato de texto para você poder copiar e colar facilmente.

-----

# Assistente Criativo de Pesquisa com GPT e RAG

Este projeto é uma aplicação web, construída com Streamlit e LangChain, que funciona como um assistente de IA avançado. Ele é capaz de responder a perguntas utilizando duas fontes de informação: uma base de conhecimento local composta por seus próprios arquivos (PDFs, DOCX, PPTX) e acesso em tempo real à internet.

O agente foi projetado para ser detalhista, criativo e fornecer respostas completas, combinando o melhor dos dois mundos: a especificidade dos seus documentos e a amplitude da web.

## 🚀 Funcionalidades Principais

  - **Modelo de IA:** Utiliza o `gpt-3.5-turbo` da OpenAI para geração de texto, otimizado para velocidade e qualidade.
  - **Base de Conhecimento Local (RAG):** Lê e interpreta arquivos `.pdf`, `.docx`, e `.pptx` localizados em uma pasta `docs`. O agente busca recursivamente em todas as subpastas.
  - **Conexão com a Internet:** Integrado com a API da Tavily para realizar buscas na web e obter informações atualizadas.
  - **Agente Inteligente:** Em vez de uma busca simples, o sistema utiliza um **Agente LangChain** que decide dinamicamente qual a melhor ferramenta usar (busca local nos documentos ou busca na web) para responder a cada pergunta.
  - **Respostas Detalhadas e Criativas:** O prompt do sistema foi cuidadosamente elaborado para instruir a IA a fornecer respostas longas, bem estruturadas, analíticas e com insights criativos.
  - **Interface Web Amigável:** Uma interface de chat limpa e intuitiva criada com Streamlit, com elementos visuais personalizados.

## ⚙️ Como Funciona (Arquitetura)

O projeto combina a arquitetura **RAG (Retrieval-Augmented Generation)** com um **Agente Autônomo**.

1.  **Ingestão de Documentos:** Ao iniciar, a aplicação escaneia a pasta `docs` e todas as suas subpastas. Os arquivos encontrados são carregados, divididos em pequenos pedaços ("chunks") e convertidos em vetores numéricos (embeddings) pelo modelo da OpenAI.
2.  **Armazenamento Vetorial:** Esses vetores são armazenados em uma base de dados vetorial em memória (FAISS), que permite buscas por similaridade semântica de forma extremamente rápida.
3.  **Lógica do Agente:** Quando o usuário faz uma pergunta, o Agente LangChain é ativado. Ele tem acesso a duas ferramentas:
      - `document_search`: Para buscar informações na base vetorial (FAISS).
      - `tavily_search_results_json`: Para pesquisar na internet.
4.  **Tomada de Decisão:** Com base na pergunta e nas instruções do prompt do sistema, o agente decide qual ferramenta usar. Ele foi instruído a **priorizar** a busca nos documentos locais. Se não encontrar uma resposta satisfatória, ele recorre à internet. Ele também pode combinar informações de ambas as fontes.
5.  **Geração da Resposta:** O GPT-3.5 Turbo recebe o contexto encontrado pelas ferramentas e gera uma resposta detalhada e criativa para o usuário.

## 🛠️ Configuração e Instalação

Siga os passos abaixo para executar o projeto localmente.

### Pré-requisitos

  - Python 3.8 ou superior
  - Pip (gerenciador de pacotes do Python)

### 1\. Crie um Ambiente Virtual (Recomendado)

```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

### 2\. Instale as Dependências

Crie um arquivo chamado `requirements.txt` com o seguinte conteúdo:

```txt
streamlit
langchain
langchain-openai
langchain-tavily
faiss-cpu
pypdf
python-docx
python-pptx
unstructured
python-dotenv
tavily-python
```

Em seguida, instale todas as bibliotecas de uma vez:

```bash
pip install -r requirements.txt
```

### 3\. Configure as Chaves de API

Você precisará de chaves de API da OpenAI e da Tavily.

  - **OpenAI:** [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
  - **Tavily:** [app.tavily.com/home](https://app.tavily.com/home)

Crie um arquivo chamado `.env` na raiz do seu projeto e adicione suas chaves:

```
OPENAI_API_KEY="sk-..."
TAVILY_API_KEY="tvly-..."
```

### 4\. Adicione seus Documentos

Crie uma pasta chamada `docs` na raiz do seu projeto. Coloque todos os seus arquivos `.pdf`, `.docx` e `.pptx` dentro desta pasta. Você pode organizar os arquivos em subpastas, pois o agente buscará em todas elas.

## ▶️ Como Executar a Aplicação

Com o ambiente virtual ativado e as configurações prontas, execute o seguinte comando no seu terminal:

```bash
streamlit run app.py
```

Seu navegador será aberto com a interface do assistente de pesquisa.

## ☁️ Publicando na Internet (Deploy)

Você pode publicar este agente gratuitamente na **Streamlit Cloud**.

1.  **Envie seu projeto para o GitHub:** Crie um repositório no GitHub e envie todos os seus arquivos, incluindo `app.py`, `requirements.txt` e a pasta `docs` com seus documentos. **NÃO envie o arquivo `.env`**.
2.  **Crie uma conta no Streamlit Cloud:** Acesse [share.streamlit.io](https://share.streamlit.io/).
3.  **Faça o Deploy:** Conecte sua conta do GitHub, selecione o repositório e clique em "Deploy".
4.  **Configure os "Secrets":** Nas configurações avançadas do seu aplicativo no Streamlit Cloud, adicione suas chaves de API (`OPENAI_API_KEY` e `TAVILY_API_KEY`). Isso garante que elas fiquem seguras.
