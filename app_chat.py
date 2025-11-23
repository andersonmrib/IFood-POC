import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
CHROMA_PATH = "./chroma_db_ifood"

def main():

    # Verifica se o banco existe
    if not os.path.exists(CHROMA_PATH):
        print("❌ Erro: Banco de dados não encontrado. Rode o ingest_csv.py primeiro.")
        return

    print("--- 🤖 Agente de Suporte Interno iFood (RAG Ativo) ---")

    # 2. Carrega o Banco de Dados (A Memória)
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # 3. Configura o Cérebro (GPT-4o-mini é rápido e barato)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 4. O PROMPT (Aqui aplicamos as regras de negócio que você pediu)
    template = """
    Você é um agente interno do IFood que auxilia colaboradores a decidirem sobre reembolsos e cancelamentos.
    
    Sua missão:
    1. Use APENAS o contexto abaixo (regras e políticas) para responder.
    2. Sempre consulte a base de conhecimento antes de responder.
    3. Se não houver confiança suficiente, sugira validação manual ou abertura de ticket interno, em vez de gerar uma resposta incerta.
    
    Contexto recuperado da Base de Conhecimento:
    {context}

    Pergunta do Colaborador: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    # 5. Loop de Conversa
    print("✅ Sistema pronto. Pergunte sobre regras de reembolso, cancelamento, etc.")
    print("(Digite 'sair' para encerrar)\n")
    
    while True:
        pergunta = input("👤 Você: ")
        if pergunta.lower() in ["sair", "exit", "tchau"]:
            break

        print("🔍 Consultando base de conhecimento...")
        
        # --- RAG MANUAL (Passo a Passo) ---
        
        # A. Busca os 3 pedaços de regras mais relevantes
        docs = vector_store.similarity_search(pergunta, k=3)
        
        # B. Monta o texto de contexto
        contexto_texto = "\n\n".join([d.page_content for d in docs])
        
        # (Debug: Mostra quais fontes ele achou. 
        fontes = set([d.metadata.get('source', 'Desconhecido') for d in docs])
        
        # C. Gera a resposta
        chain = prompt | llm | StrOutputParser()
        resposta = chain.invoke({
            "context": contexto_texto,
            "question": pergunta
        })
        
        # D. Exibe a resposta final
        print(f"\n🤖 iFood AI: {resposta}")
        print("-" * 50)
        print(f"   [Fontes consultadas: {', '.join(fontes)}]")
        print("-" * 50)

if __name__ == "__main__":
    main()