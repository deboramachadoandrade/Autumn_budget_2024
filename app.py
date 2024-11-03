# Perlon AI User Chatbot
# Debora Andrade, 01.11.2024

# This chatbot is built using Streamlit as UI and LangChain for creating an agent and managing memory


import streamlit as st
import asyncio
import os
from openai import AsyncOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.agents import create_openai_functions_agent, OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder
from dotenv import load_dotenv



load_dotenv()



async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key)

    st.set_page_config(
        page_title="UK Autumn Budget 2024",
        page_icon="",
        layout="wide",
    )



    "# Your 2024 Autumn Budget assistant"

    @st.cache_resource(ttl="1h")

    # Here we store all the documents that we would like our chatbot to have access to: (RAG)
    # Currently there is only a single PDF containing information about Perlon AI (the info contained in the website), so the chatbot can fluently answer questions about 
    # Perlon AI and the services it offers, should this topic arise in the conversation
    def retriever():
        loader = PyMuPDFLoader("data/Autumn_Budget_2024__print_ready_.pdf") 
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
        )
        documents = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        return vectorstore.as_retriever(search_kwargs={"k": 4})

    # We give the agentic chatbot access to the RAG:
    tool = create_retriever_tool(
        retriever(),
        "search_Autumn_Budget",
        "Searches and returns information regarding the 2024 Autumn Budget. You do not know anything about the 2024 Autumn Budget, so if you are ever asked about it you should use this tool.",
    )
    tools = [tool]
    llm = ChatOpenAI(temperature=0.1, streaming=True, model="gpt-3.5-turbo")

    # Prompt that will guide the chatbot to ask relevant questions to the user:
    message = SystemMessage(
        content=(
        """On the 30th of October 2024 the newly-elected Labour government announced the 2024 Autumn Budget, one that represents a tipping point in the UK finantial history. 
The public in general is aprehensive as to how the new budget will affect their lives, and it is natural that they are interested in how their specific situation will be impacted.
Your job is to have a conversation with the user to understand their interests and concerns regarding the new budget. 
You will proactively ask questions, in a conversational fashion. 
You will gently ask the following questions (if they have not already being asked), and that will guide you to best assist the user and answer their queries:

1) Which aspect of the budget is of interest to them?
2) Would they like to know how their particular area of interest compares with previous tax scenarios?
3) Are they also interested in other aspects of the budget (e.g. inheritance tax, employer national insurance contribution, employee income tax thresholds, etc)? 

You will only ask one question at a time. Your follow-up questions will depend on the user feedback. 
It is better to understand the interests of the user well before rushing to provide an immediate answer. Therefore, you might follow up with another question or with a proper answer, depending on the situation.
If you ask something and the user answers in a way that doesn't sound clear to you, ask the same question in a different way. 
Never repeat questions literally, always bring a different light into each rephrased question, should you need to ask something again.

Unless otherwise explicitly stated, it is probably fair to assume that questions asked by the user are about the 2024 Autumn Budget.
when initial questions are answered, ask whether you can assist them with something else regarding the 2024 Autumn budget. If the user replies "no", wish them a good day, otherwise keep answering their questions.

If the user asks a question that falls outside the scope of the new budget, answer that you were instructed to limit your answers to topics related to the 2024 Autumn budget.

Make sure that your answers are strictly backed by the retrieved context and that no assumptions are made. For example, if asked who is the chancellor of the Exchequer, it would be wrong to answer James Murray based on the assumption that he presented the document. James Murray is the Exchequer secretary.

Let us have an engaging conversation with the user:
"""
        )
    )
    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
    )
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
    )

    # LangChain built-in session-based memory buffer:
    memory = AgentTokenBufferMemory(llm=llm)

    starter_message = "Hello! I am here to help you understand how the Autumn Budget may impact you and your business. Which aspect of the budget are you primarily interested in?"

    # session_state will be useful for keeping certain variables through the reruns of async functions:
    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [AIMessage(content=starter_message)]

    if "original_excerpts" not in st.session_state:
        st.session_state["original_excerpts"] = ""

   




    for msg in st.session_state.messages:
        if isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        memory.chat_memory.add_message(msg)

    if prompt := st.chat_input(placeholder=starter_message):
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            response = agent_executor.invoke(
                {"input": prompt, "history": st.session_state.messages},
                callbacks=[st_callback],
                include_run_info=False,
            )
            st.session_state.messages.append(AIMessage(content=response["output"]))
            st.write(response["output"])
            
            st.sidebar.header("Excerpts from the Autumn Budget related to your question :")

            try:
                st.session_state["original_excerpts"] = response["intermediate_steps"][0][1]
                st.sidebar.write(st.session_state["original_excerpts"])
            except:
                st.sidebar.write("The original text was not used to produce this answer.")
            
            
                
            response["intermediate_steps"] = []
            memory.save_context(
                    {"input": prompt}, response
                )

            st.session_state["messages"] = memory.buffer


           
                

            

if __name__ == "__main__":
    asyncio.run(main())