from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from langchain_core.runnables import RunnableConfig
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

modelo = ChatOpenAI(
  model="gpt-4o-mini",
  temperature=0.5,
  api_key=api_key
)

prompt_consultor_praia = ChatPromptTemplate.from_messages(
   [
     ("system", "Voce e um consultor de viagens para a praia"),
     ("human", "{query}")
   ]
 )

prompt_consultor_montanha = ChatPromptTemplate.from_messages(
   [
     ("system", "Voce e um consultor de viagens para a montanha"),
     ("human", "{query}")
   ]
 )

prompt_roteador = ChatPromptTemplate.from_messages(
   [
     ("system", "Apenas resposta com 'praia' ou 'montanha'"),
     ("human", "{query}")
   ]
 )
 
class Rota(TypedDict):
  destino: Literal["praia", "montanha"]

class Estado(TypedDict):
  query: str
  destino: Rota
  resposta: str

cadeia_praia = prompt_consultor_praia | modelo | StrOutputParser()
cadeia_montanha = prompt_consultor_montanha | modelo | StrOutputParser()
roteador = prompt_roteador | modelo.with_structured_output(Rota)

async def no_roteador(estado: Estado, config=RunnableConfig):
  return {"destino": await roteador.ainvoke({"query": estado["query"]}, config)}

async def no_praia(estado: Estado, config=RunnableConfig):
  return {"resposta": await cadeia_praia.ainvoke({"query": estado["query"]}, config)}

async def no_montanha(estado: Estado, config=RunnableConfig):
  return {"resposta": await cadeia_montanha.ainvoke({"query": estado["query"]}, config)}

def escolher_no(estado: Estado) -> Literal["praia", "montanha"]:
  return "praia" if estado["destino"]["destino"] == "praia" else "montanha"

grafo = StateGraph(Estado)
grafo.add_node("rotear", no_roteador)
grafo.add_node("praia", no_praia)
grafo.add_node("montanha", no_montanha)
grafo.add_edge(START, "rotear")
grafo.add_conditional_edges("rotear", escolher_no)
grafo.add_edge("praia", END)
grafo.add_edge("montanha", END)

app = grafo.compile()

async def main():
  resposta = await app.ainvoke({"query": "Quero uma jurisprudencia pequena simples."})
  print(resposta["resposta"])

asyncio.run(main())