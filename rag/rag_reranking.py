from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import  PromptTemplate, ChatPromptTemplate
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")

PERSIST_DIRECTORY='./chroma-rh'

def carrega_documentos():
  caminhos_documentos = [
    "rag/documentos/codigo_conduta.pdf",
    "rag/documentos/politica_ferias.pdf",
    "rag/documentos/politica_home_office.pdf"
  ]

  documentos = []

  for caminho in caminhos_documentos:
    loader = PyPDFLoader(caminho)

    docs = loader.load()

    for doc in docs:
      doc.metadata["medicamento"] = caminho.split("/")[-1].replace(".pdf", "")

    documentos.extend(docs)
  return documentos
  
def gera_chunks(documentos):
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=150
  )

  chunks = text_splitter.split_documents(documentos)
  return chunks

def aplica_metadados(chunks):
  for chunk in chunks:
    texto = chunk.page_content.lower()

    if "férias" in texto:
        chunk.metadata["categoria"] = "ferias"
    elif "home office" in texto or "remoto" in texto:
        chunk.metadata["categoria"] = "home_office"
    elif "conduta" in texto or "ética" in texto:
        chunk.metadata["categoria"] = "conduta"
    else:
        chunk.metadata["categoria"] = "geral"

  return chunks

def cria_vectorstore(chunks):
  embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
  )

  vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIRECTORY
  )

  return vectorstore

def rerank_documentos(pergunta, documentos, llm):
    """
    Reordena os documentos recuperados com base na relevância
    usando o próprio LLM (reranking semântico)
    """

    prompt_rerank = PromptTemplate(
        input_variables=["pergunta", "texto"],
        template="""
Você é um especialista em políticas internas de RH.

Pergunta do usuário:
{pergunta}

Trecho do documento:
{texto}

Avalie a relevância desse trecho para responder a pergunta.
Responda apenas com um número de 0 a 10.
"""
    )

    documentos_com_score = []

    for doc in documentos:
        score = llm.invoke(
            prompt_rerank.format(
                pergunta=pergunta,
                texto=doc.page_content
            )
        ).content

        try:
            score = float(score)
        except:
            score = 0

        documentos_com_score.append((score, doc))

    # Ordena do mais relevante para o menos relevante
    documentos_ordenados = sorted(
        documentos_com_score,
        key=lambda x: x[0],
        reverse=True
    )

    # Retorna apenas os documentos
    return [doc for _, doc in documentos_ordenados]

def responder_pergunta(pergunta, vectorstore):
  llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=api_key
  )

  documentos_recuperados = vectorstore.similarity_search(
    pergunta,
    k=8
  )

  documentos_rerankeados = rerank_documentos(pergunta, documentos_recuperados, llm)

  contexto_final = documentos_rerankeados[:4]

  contexto_texto = "\n\n".join(
     [doc.page_content for doc in contexto_final]
  )

  prompt = """
  Você é um agente de RH corporativo.
  Responda APENAS com base nas políticas internas abaixo.

  Contexto:
  {contexto_texto}

  Pergunta:
  {pergunta}
    """

  prompt_formatado = prompt.format(
    contexto_texto=contexto_texto,
    pergunta=pergunta
  )

  resposta = llm.invoke(prompt_formatado)
  return resposta.content, contexto_final

# Fluxo de processamento
# documentos = carrega_documentos()
# chunks = gera_chunks(documentos)
# chunks_metadados = aplica_metadados(chunks)
# vectorstore = cria_vectorstore(chunks_metadados)

# resposta, fontes = responder_pergunta("Quem pode trabalhar em regime de home office e quais são as condições?", vectorstore)

# print("Resposta: ", resposta)
# print("Fontes: ", fontes)

# INTERFACE STREAMLIT
st.set_page_config(page_title="Agente de RH com RAG", layout="wide")
st.title("🤖 Agente de RH — Políticas Internas")

pergunta = st.text_input("Digite sua pergunta sobre políticas internas de RH:")

if pergunta:
    with st.spinner("Consultando políticas internas..."):
        documentos = carrega_documentos()
        chunks = gera_chunks(documentos)
        chunks = aplica_metadados(chunks)
        vectorstore = cria_vectorstore(chunks)

        resposta, fontes = responder_pergunta(pergunta, vectorstore)

    st.subheader("Resposta")
    st.write(resposta)

    st.subheader("Fontes utilizadas")
    for i, doc in enumerate(fontes, start=1):
        st.markdown(f"**Trecho {i}**")
        st.write(f"Documento: {doc.metadata.get('documento')}")
        st.write(f"Categoria: {doc.metadata.get('categoria')}")
        st.write(doc.page_content)
        st.divider()

# rodar com uv run streamlit run rag/rag_reranking.py