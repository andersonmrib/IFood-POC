# 🤖 [IFood - POC] - RAG Agent

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.137-green?style=flat-square)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4.6-orange?style=flat-square)](https://www.trychroma.com/)

**[Sistema para consulta de políticas do estabelecimento]**

---

## 📋 Sobre o Projeto

O **[IFood - POC]** é uma aplicação que utiliza **Retrieval-Augmented Generation (RAG)** para analisar políticas do estabelecimento. A solução combina busca semântica com IA generativa (OpenAI) para fornecer respostas precisas para as dúvidas dos clientes.

## 🚀 Instalação

### Pré-requisitos
* Python 3.12+
* Chave de API da OpenAI
* Gerenciador de pacotes `pip`

### 1: Clonar o Repositório
```bash
git clone <url-do-seu-repositorio>
cd <nome-da-pasta>
```

### 2: Configurar Ambiente Virtual
```bash
python -m venv venv
# Linux/Mac
source venv/bin/activate  
# Windows
venv\Scripts\activate  
```

### 3: Instalar Dependências
```bash
pip install -r requirements.txt
```

### 4: Configurar Variáveis de Ambiente
Crie um arquivo `.env` na raiz do projeto e insira sua chave da OpenAI:
```env
OPENAI_KEY=sk-sua-chave-aqui
```

---

## 💻 Como Utilizar

### 1. Ingestão de Dados (Vector DB)
Processe o documento de políticas para criar a base de conhecimento da IA:
```bash
python ingestion.py
```

### 2. Iniciar a Interface
Execute o dashboard do Streamlit:
```bash
streamlit run app.py
```
Acesse no seu navegador.


## 📄 Licença
Este projeto é uma Prova de Conceito (POC) para fins de demonstração técnica e estudo.
