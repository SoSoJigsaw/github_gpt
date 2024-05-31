import os
import git
import re
import openai
import requests
import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv

system_prompt = """Você é um programador experiente especialista em criar documentação técnica no github usando 
markdown (arquivos .md). Por isso, você está me ajudando a criar meu portfólio, seguindo a seguinte estrutura de 
exemplo para documentar meus projetos realizados na faculdade: 
´´´# Meus Projetos 
#### Em 2019-2 
Trabalhei no projeto da API com o Parceiro Acadêmico... {...}. <br> Percebemos, cada vez mais... {...} <br> 
[link para o GIT](https://link-para-o-repositorio-no-github.com) 
#### Tecnologias Utilizadas 
## Contribuições Pessoais 
O que temos que ter sempre em mente é que... {...} 
#### Hard Skills Efetivamente Desenvolvidas
- (Exemplo)Java: {...} 
- (Exemplo) SQL: {...} 
- (Exemplo)JavaScript: {...} 
#### Soft Skills Efetivamente Desenvolvidas 
- (Exemplo)Autonomia: {...} <hr>  
#### Em 2020-1 
Trabalhei no projeto da API com o Parceiro Acadêmico... {...}. <br> O incentivo ao avanço tecnológico, assim 
como a preocupação com... {...} <br> [link para o GIT](https://link-para-o-repositorio-no-github.com) 
#### Tecnologias Utilizadas 
- (Exemplo)RPA: {...}  
## Contribuições Pessoais 
Contribui no projeto com relação ao sistema de... {...} 
#### Hard Skills Efetivamente Desenvolvidas 
- (Exemplo)Java: {...} 
- (Exemplo) SQL: {...} 
- (Exemplo)JavaScript: {...} 
#### Soft Skills Efetivamente Desenvolvidas 
- (Exemplo)Proatividade: {...} <hr>´´´

Resumindo, o seu "goal" é: Utilizando apenas informações fornecidas a você, gere documentos em formato markdown para 
serem colocados no github em arquivos .md, seguindo a estrutura passada anteriormente acima. Os arquivos que você terá  
acesso pelo banco vetorizado, são arquivos do repositório de um dos projetos acadêmicos que eu realizei e que preciso 
realizar a sua documentação no formato fornecido a você anteriormente, necessariamente em markdown. Eu desejo que você 
analise os arquivos que lhe enviei para conhecimento e que você gere um texto markdown seguindo a estrutura de 
documentação da minha faculdade passada anteriormente para você. Analise, extraia informações relevantes e gere a 
documentação em markdown. Responda somente com a documentação gerada em formato de markdown. Não inclua texto antes ou 
depois, o output deve ser apenas a documentação necessariamente em markdown. Os locais onde coloquei "{...}" devem ser
preenchidos e desenvolvidos por você, justificando cada ponto considerado em cada parte da documentação, da forma mais 
técnica, explicativa, desenvolvida, não deixando de lado nenhum argumento relevante. Então por favor, tente desenvolver 
o texto sempre tentando ser o mais detalhado possível e justificando tudo que é apontado, mas sem citar em suas 
resposta a cópia de trechos do código."""


# Função para clonar o repositório do GitHub
def clone_repository(repo_url, clone_dir):
    if not os.path.exists(clone_dir):
        os.makedirs(clone_dir)
    git.Repo.clone_from(repo_url, clone_dir)
    st.success(f"Repositório clonado em: {clone_dir}")


# Função para ler arquivos do diretório
def read_files_from_directory(directory):
    file_contents = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                file_contents.append((file_path, content))
    return file_contents


# Função de pré-processamento
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text).strip()  # Removendo espaços extras
    return text


# Função para gerar embeddings com OpenAI
def get_embeddings(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']


# Função para armazenar embeddings em FAISS
def store_embeddings_in_faiss(embeddings):
    if not embeddings:
        st.error("Nenhum embedding foi gerado.")
        return None, None

    dimension = len(embeddings[0][1])
    index = faiss.IndexFlatL2(dimension)
    meta_data = []
    for path, embedding in embeddings:
        index.add(np.array([embedding], dtype=np.float32))
        meta_data.append(path)
    return index, meta_data


# Função para extrair metadados do repositório
def get_repo_metadata(repo_url):
    repo_name = repo_url.split('/')[-1]
    user_name = repo_url.split('/')[-2]
    api_url = f"https://api.github.com/repos/{user_name}/{repo_name}"
    response = requests.get(api_url)
    repo_data = response.json()

    metadata = {
        "full_name": repo_data.get("full_name"),
        "description": repo_data.get("description"),
        "language": repo_data.get("language"),
        "stargazers_count": repo_data.get("stargazers_count"),
        "forks_count": repo_data.get("forks_count"),
        "open_issues_count": repo_data.get("open_issues_count"),
        "created_at": repo_data.get("created_at"),
        "updated_at": repo_data.get("updated_at")
    }

    return metadata


# Função para recuperar documentos similares
def retrieve_similar_documents(query, index, metadata, top_k=5):
    query_embedding = get_embeddings(query)
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), top_k)
    results = [(metadata[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    return results


# Função para construir o contexto com chunk overlapping
def build_context_with_chunking(documents, max_length=2000, overlap=200):
    context = ""
    for doc, _ in documents:
        if len(context) + len(doc) <= max_length:
            context += doc + " "
        else:
            break
    return context[:max_length]


# Função para gerar resposta do ChatGPT com role system
def generate_response(query, context, system_prompt = system_prompt):

    prompt = f"{system_prompt}\n\nContexto: {context}\n\nQuery do usuário: {query}\nAI:"

    response = openai.Completion.create(
        model="gpt-4o-2024-05-13",
        prompt=prompt
    )
    return response.choices[0].text.strip()


# Função principal para integrar tudo
def chat_with_custom_gpt(query, index, metadata):
    similar_documents = retrieve_similar_documents(query, index, metadata)
    context = build_context_with_chunking(similar_documents)
    response = generate_response(query, context)
    return response


# Interface do Streamlit
def main():

    load_dotenv()

    st.title("Custom GPT Chat for GitHub Repository Analysis")

    openai.api_key = os.getenv("OPENAI_API_KEY")

    if not openai.api_key:
        st.warning("Por favor, configure sua chave de API do OpenAI no arquivo .env antes de prosseguir.")
        st.stop()

    repo_url = st.text_input("Digite a URL do repositório do GitHub:")
    clone_dir = "./cloned_repo"

    if st.button("Clonar Repositório"):
        clone_repository(repo_url, clone_dir)

    file_contents = read_files_from_directory(clone_dir)

    if file_contents:
        preprocessed_contents = [(path, preprocess_text(content)) for path, content in file_contents]
        embeddings = [(path, get_embeddings(content)) for path, content in preprocessed_contents if content.strip()]

        index, meta_data = store_embeddings_in_faiss(embeddings)

        if index and meta_data:
            repo_metadata = get_repo_metadata(repo_url)
            st.write("Metadados do repositório:", repo_metadata)

            query = st.text_input("Digite sua consulta:")

            if st.button("Enviar Consulta"):
                response = chat_with_custom_gpt(query, index, meta_data)
                st.write("Resposta do GPT:", response)
        else:
            st.error("Erro ao criar o índice de embeddings.")
    else:
        st.error("Nenhum arquivo foi lido do repositório.")


if __name__ == "__main__":
    main()  # Executar em um terminal: streamlit run app.py

