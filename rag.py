
import os
os.environ["GOOGLE_API_KEY"] = ".........................."

# %%
from langchain_google_genai import GoogleGenerativeAI
llm = GoogleGenerativeAI(model="gemini-1.5-flash")

response = llm.invoke("What are some of the pros and cons of Python as a programming language?")
print(response)

# %%
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a English french translator. You return whatever the user says in french"),
    ("user", "{input}")
])

chain = prompt | llm

chain.invoke({"input":"I enjoy going to rock concert!"})

# %%
from langchain_core.output_parsers import StrOutputParser

output_parser =StrOutputParser()

chain = prompt | llm | output_parser

chain.invoke({"input":"I enjoy going to rock concert!"})

# %%
llm.invoke("what do you know about langchain?")

# %%
llm.invoke("Do you know what happend in 5th august,2024 in Bangladesh?")
# %%
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://python.langchain.com/docs/introduction")

docs = loader.load()

# %%
docs

# %%
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# %%
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

vectorstore = FAISS.from_documents(documents, embeddings)

# %%
from langchain.chains.combine_documents import create_stuff_documents_chain
template = """Answer the following question based only on the provided context:
<context>
{context}
</context>

Question: {input}
"""
prompt = ChatPromptTemplate.from_template(template)
document_chain = create_stuff_documents_chain(llm, prompt)

# %%
from langchain_core.documents import Document

document_chain.invoke({
    "input": "what is langchain?",
    "context": [Document(page_content="LangChain is a framework for developing applications powered by large language models (LLMs).")]
})

# %%
from langchain.chains import create_retrieval_chain

retriever = vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# %%
response = retrieval_chain.invoke({
    "input":"what is langchain all about?"
})

# %%
response['answer']

# %%
# conversational retrieval chain

from langchain.chains import create_history_aware_retriever

from langchain_core.prompts import MessagesPlaceholder

prompt = ChatPromptTemplate([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "given the avobe conversation, generate a search query to look up in order to get information relevent to the conversation")
])

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# %%
from langchain_core.messages import HumanMessage, AIMessage

chat_history = [
    HumanMessage(content="Is there anything new about langchain?"),
    AIMessage(content="YES")
]

response = retrieval_chain.invoke({
    "chat_history":chat_history,
    "input":"Tell me more about it!"
})

response["answer"]

# %%
from langchain.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the users questions based on the below context: \n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

document_chain = create_stuff_documents_chain(llm, prompt)
conversational_retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

# %%
response = conversational_retrieval_chain.invoke({
    "chat_history": [],
    "input": "What is langchain?"
})

response["answer"]
