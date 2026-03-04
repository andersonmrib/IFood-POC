🚀 Como Instalar e Rodar o Projeto
Este guia auxiliará na configuração do ambiente local para execução do Agente de IA com busca semântica.

1. Pré-requisitos
Certifique-se de ter o Python 3.9+ instalado em sua máquina.

2. Clonar o Repositório
git clone https://github.com/andersonmrib/IFood-POC
cd nome-do-seu-repo

3. Configurar o Ambiente Virtual
É altamente recomendável o uso de um ambiente isolado:
python -m venv venv
# No Windows:
.\venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate

4. Instalar Dependências
pip install -r requirements.txt

5. Configurar Variáveis de Ambiente
Crie um arquivo .env na raiz do projeto e adicione sua chave da API:
OPENAI_API_KEY=sua_chave_aqui

6. Executar a Aplicação
O projeto utiliza Streamlit para a interface. Para iniciar:
streamlit run app.py
