import streamlit as st #Python -m streamlit run app_gui.py
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from PIL import Image

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Suporte interno do IFood",
    page_icon="🍔",
    layout="centered",
    initial_sidebar_state='expanded'
)

# Carrega variáveis de ambiente (.env)
load_dotenv()
CHROMA_PATH = "./chroma_db_ifood"
LOGO_PATH = "logo_ifood.png"

# --- 2. ESTILIZAÇÃO CSS (iFood Style) ---
st.markdown("""
<style>
            
    /* Fundo principal */
    .stApp {
        background-color: #F7F7F7;
    }
    
    /* Cabeçalho fixo */
    .header-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
            
    /* Estilo da Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F7F7F7;
        border-right: 1px solid #E0E0E0;
    }
    
    /* Chat Bubbles - Container */
    .stChatMessage {
        background-color: white;
        border-radius: 15px;
        border: 1px solid #E0E0E0;
    }

    /* Força a cor preta (#000000) em todos os elementos de texto dentro do chat */
    .stChatMessage div[data-testid="stMarkdownContainer"] p, 
    .stChatMessage div[data-testid="stMarkdownContainer"] li,
    .stChatMessage div[data-testid="stMarkdownContainer"] span,
    .stChatMessage div[data-testid="stMarkdownContainer"] {
        color: #000000 !important;
    }
    /* ------------------------------------ */
    
    /* Botão de Enviar e Inputs */
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
            
    .stButton > button {
        width: 100%;
        border: none;
        background-color: #EA1D2C !important;
        color: white !important;
        /* Transição suave para cor e fundo */
        transition: all 0.3s ease !important;
    }
    
    /* Estado Hover (Passando o mouse) */
    .stButton > button:hover {
        background-color: #C21824 !important;
        color: white !important;
        transform: scale(1.01); /* Leve efeito de zoom */
    }

    /* Estado Active (Clicando) e Focus - CORRIGE O PISCAR BRANCO */
    .stButton > button:active, .stButton > button:focus {
        background-color: #EA1D2C !important; /* Mantém vermelho */
        color: black !important;               /* Texto preto ao clicar */
        border: none !important;
        box-shadow: none !important;           /* Remove borda de foco padrão */
        outline: none !important;
    }   
    
    /* Links na cor do iFood */
    a {
        color: #EA1D2C !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. CABEÇALHO VISUAL ---
# Se a imagem não aparecer, aparecerá o texto "Logo iFood"
st.markdown("""
<div class="header-container">
    <h3 style='color: #333; margin: 0; font-family: sans-serif; font-weight: bold;'>Suporte GenAI Interno</h3>
    <p style='color: #666; font-size: 14px; margin-top: 5px;'>Assistente de Políticas de Reembolso e Cancelamento</p>
</div>
""", unsafe_allow_html=True)

def load_image(image_path):
    if(os.path.exists(image_path)):
        return Image.open(image_path)
    return None

logo_img = load_image(LOGO_PATH)
# --- 4. Sidebar ---

with st.sidebar:
    if logo_img:
        st.image(logo_img, width=200)
    else:
        st.markdown("## 🍔 iFood GenAI")
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("Painel do Agente")
    st.info("""
    Este assistente utiliza IA para buscar regras nas políticas internas.
    
    **Como usar:**
    1. Digite a dúvida do colaborador.
    2. Analise a resposta e as fontes.
    3. Em caso de dúvida, siga o fluxo de ticket.
    """)
    
    st.write("---")
    
    # Botão de Limpar Histórico (Estilizado pelo CSS acima)
    if st.button("🗑️ Nova Consulta"):
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Olá! Estou pronto para ajudar com a próxima dúvida sobre políticas."}
        ]
        st.rerun()

    st.write("---")
    st.caption("🔒 POC v1.2 | Uso Interno Confidencial")

# --- 4. CARREGAMENTO DO CÉREBRO (RAG) ---
@st.cache_resource
def load_rag_chain():
    # Verifica se o banco existe
    if not os.path.exists(CHROMA_PATH):
        st.error("❌ Erro Crítico: O banco de dados não foi encontrado.")
        st.info("Por favor, execute o arquivo `ingest_csv.py` primeiro para criar a base de conhecimento.")
        return None, None

    # Carrega Embeddings e Banco Vetorial
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    
    # Carrega o LLM (GPT-4o-mini para rapidez e baixo custo)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Define o Prompt com as regras de negócio
    template = """
    Você é um agente interno que auxilia colaboradores a decidirem sobre reembolsos e cancelamentos.
    Sempre consulte a base de conhecimento abaixo antes de responder.
    Se não houver confiança suficiente, sugira validação manual ou abertura de ticket interno.

    Contexto recuperado:
    {context}

    Pergunta: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    return vector_store, prompt, llm

# Inicializa os componentes
vector_store, prompt, llm = load_rag_chain()

# Para a execução se o banco não carregar
if vector_store is None:
    st.stop()

# --- 5. GERENCIAMENTO DE HISTÓRICO ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Olá! Sou o assistente de políticas internas. Qual a sua dúvida sobre o processo?"}
    ]

# Renderiza mensagens antigas
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user", avatar="👤").write(msg["content"])
    else:
        st.chat_message("assistant", avatar="🤖").write(msg["content"])

# --- 6. LOOP DE INTERAÇÃO (CHAT) ---
if question := st.chat_input("Digite sua dúvida (ex: cliente pediu reembolso após sair para entrega)..."):
    
    # A. Exibe pergunta do usuário
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user", avatar="👤").write(question)

    # B. Gera a resposta
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Consultando as regras..."):
            
            # Passo 1: Busca Semântica (Recupera 3 documentos)
            docs = vector_store.similarity_search(question, k=3)
            
            # Passo 2: Prepara o contexto
            context_text = "\n\n".join([d.page_content for d in docs])
            
            # (Opcional) Extrai as fontes para exibir no rodapé da resposta
            fontes = sorted(list(set([d.metadata.get('source', 'Desconhecido') for d in docs])))
            
            # Passo 3: Envia para o GPT
            chain = prompt | llm | StrOutputParser()
            response = chain.invoke({
                "context": context_text,
                "question": question
            })
            
            # Passo 4: Exibe o resultado
            st.write(response)
            
            # Mostra as fontes de forma discreta
            if fontes:
                st.caption(f"📚 **Fontes consultadas:** {', '.join(fontes)}")
    
    # C. Salva resposta no histórico
    st.session_state.messages.append({"role": "assistant", "content": response})

# Rodapé simples
st.markdown("<br><hr><center><small style='color: grey'>POC iFood GenAI - Uso Interno Confidencial</small></center>", unsafe_allow_html=True)