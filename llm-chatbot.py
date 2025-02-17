# %%
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# %%
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq

model = ChatGroq(model="llama3-8b-8192")

# %%
from langchain_core.messages import HumanMessage

model.invoke([HumanMessage(content="Hi! I'm Shanto")])

# %%
model.invoke([HumanMessage(content="What's my Name?")])

# %%
from langchain_core.messages import AIMessage

model.invoke([
    HumanMessage(content="Hi! I'm Shanto"),
    AIMessage(content="Hello Shanto! How can i assist you today?"),
    HumanMessage(content="what's my name?")
])

# %%
from langchain_core.chat_history import (BaseChatMessageHistory, InMemoryChatMessageHistory)

from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id : str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(model, get_session_history)
print(with_message_history)

# %%
config = {"configurable":{"session_id":"abc2"}}

# %%
response = with_message_history.invoke(
    [HumanMessage(content="Hello! I'm Shanto")],
    config=config
)

response.content

# %%
config = {"configurable":{"session_id":"abc3"}}

# %%
response = with_message_history.invoke(
    [HumanMessage(content="What's my Name?")],
    config=config
)
response.content

# %%
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpfull assistant. Answer all questions to the best of your ability"
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain = prompt | model

# %%
response = chain.invoke({"messages": [HumanMessage(content="Hi! I'm Shanto")]})
response.content

# %%
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpfull assistant. Answer all questions to the best of your ability in {language}"
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain = prompt | model

# %%
response = chain.invoke(
    {"messages":[HumanMessage(content="Hi! I'm Shanto!")], "language":"Italian"}
)
response.content

# %%
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages"
)

# %%
config = {"configurable" : {"session_id": "abc11"}}

# %%
response = with_message_history.invoke(
    {"messages": [HumanMessage(content="Hi! I'm cruise")], "language":"spanish"},
    config=config
)

response.content

# %%
from langchain_core.messages import SystemMessage, trim_messages

trimmer = trim_messages(
    max_tokens=30,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

trimmer.invoke(messages)

# %%
from operator import itemgetter

from langchain_core.runnables import RunnablePassthrough

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what's my name?")],
        "language": "English",
    }
)
response.content

# %%
response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what math problem did i ask?")],
        "language":"English"
    }
)
response.content

# %%
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages"
)

config = {"configurable":{"session_id":"abc20"}}

# %%
response = with_message_history.invoke(
    {
        "messages": messages + [HumanMessage(content="whats my name?")],
        "language": "English",
    },
    config=config,
)

response.content

# %%
config = {"configurable":{"session_id":"abc15"}}

# %%
for r in with_message_history.stream(
    {
        "messages":[HumanMessage(content="Hi! I'm shanto! tell me a joke")],
        "language":"English"
    },
    config=config
):print(r.content, end="|")
