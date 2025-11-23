import os
import shutil
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Carrega API
load_dotenv()

def main():
    
    csv_file = "base_conhecimento_ifood_genai-exemplo.csv"
    chroma_path = "./chroma_db_ifood"

    #1. Checa se CSV existe
    if not os.path.exists(csv_file):
        print(f"ERRO: Arquivo não encontrado {csv_file}")
        return
    
    #2. Limpa banco antigo para evitar duplicata
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
        print("Banco antigo limpo.")

    print(f"Lendo {csv_file}...")

    #3. Carrega o CSV
    loader = CSVLoader(
        file_path=csv_file,
        encoding='utf-8',
        source_column='fonte'
    )
    documents = loader.load()

    print(f"Criando embeddings para {len(documents)} regras...")

    #4. Cria banco vetorial
    Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(),
        persist_directory=chroma_path
    )

    print("Ok! Banco de dados criado!")

if __name__ == "__main__":
    main()