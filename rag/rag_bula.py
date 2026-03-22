import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA 
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")

caminhos_bulas = [
  "rag/documentos/dipirona.pdf",
  "rag/documentos/paracetamol.pdf"
]

documentos = []

for caminho in caminhos_bulas:
  loader = PyPDFLoader(caminho)

  docs = loader.load()

  for doc in docs:
    doc.metadata["medicamento"] = caminho.split("/")[-1].replace(".pdf", "")

  documentos.extend(docs)

text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=600,
  chunk_overlap=150
)

chunks = text_splitter.split_documents(documentos)

# Percorre cada chunk para classificar semanticamente seu conteúdo
for chunk in chunks:

    # Normaliza o texto para facilitar as verificações
    texto = chunk.page_content.lower()

    # Identificação do medicamento
    if "identificação do medicamento" in texto or "composição" in texto:
        chunk.metadata["categoria"] = "identificacao"

    # Indicações terapêuticas
    elif "indicação" in texto or "para que este medicamento é indicado" in texto:
        chunk.metadata["categoria"] = "indicacao"

    # Funcionamento do medicamento
    elif "como este medicamento funciona" in texto or "ação" in texto:
        chunk.metadata["categoria"] = "como_funciona"

    # Contraindicações
    elif "contraindicação" in texto or "quando não devo usar" in texto:
        chunk.metadata["categoria"] = "contraindicacao"

    # Advertências e precauções
    elif "advertência" in texto or "precaução" in texto or "o que devo saber antes de usar" in texto:
        chunk.metadata["categoria"] = "advertencias_precaucoes"

    # Interações medicamentosas
    elif "interação" in texto or "interações medicamentosas" in texto:
        chunk.metadata["categoria"] = "interacoes"

    # Posologia e modo de uso
    elif "dose" in texto or "posologia" in texto or "como devo usar" in texto:
        chunk.metadata["categoria"] = "posologia_modo_uso"

    # Reações adversas
    elif "reações adversas" in texto or "quais os males" in texto:
        chunk.metadata["categoria"] = "reacoes_adversas"

    # Armazenamento
    elif "onde, como e por quanto tempo posso guardar" in texto or "armazenar" in texto:
        chunk.metadata["categoria"] = "armazenamento"

    # Superdosagem
    elif "quantidade maior do que a indicada" in texto or "superdosagem" in texto:
        chunk.metadata["categoria"] = "superdosagem"

    # Conteúdo geral / administrativo
    else:
        chunk.metadata["categoria"] = "geral"


embeddings = OpenAIEmbeddings(
  model="text-embedding-3-small"
)

vectorstore = Chroma.from_documents(
  documents=chunks,
  embedding=embeddings,
  persist_directory="./chroma_bulas"
)
     
retriever = vectorstore.as_retriever(
  kwargs={"k":4}
)

llm = ChatOpenAI(
  model="gpt-4o-mini",
  temperature=0.5,
  api_key=api_key
)

prompt = ChatPromptTemplate.from_template("""
Responda com base no contexto: {context}
Pergunta: {input}
""")

combine_docs_chain = create_stuff_documents_chain(llm, prompt)

qa_chain = create_retrieval_chain(
  retriever,
  combine_docs_chain
)

resposta = qa_chain.invoke({"input": "Qual as contraindicacoes da composicao da dipirona"})

print("Pergunta:")
print(resposta['input'])

print("\nResposta do Agente:")
print(resposta["answer"])

print("\nTrechos utilizados como contexto:\n")

# Percorre os documentos recuperados
for i, doc in enumerate(resposta["context"], start=1):
    print(f"--- Trecho {i} ---")

#     # Metadados principais 
    print(f"Medicamento: {doc.metadata.get('medicamento', 'N/A')}")
    print(f"Categoria: {doc.metadata.get('categoria', 'N/A')}")
    print(f"Documento: {doc.metadata.get('source', 'Documento desconhecido')}")
    print(f"Página: {doc.metadata.get('page', 'N/A')}")

#     # Conteúdo recuperado
    print("\nConteúdo do chunk:")
    print(doc.page_content)
    print("\n")
