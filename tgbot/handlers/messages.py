"""
This module contains handlers that handle messages from users

Handlers:
    echo_handler    - echoes the user's message

Note:
    Handlers are imported into the __init__.py package handlers,
    where a tuple of HANDLERS is assembled for further registration in the application
"""

import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def get_vectorstore_icp(faiss_name: str, embeddings: OpenAIEmbeddings) -> FAISS:
    vectorStores = FAISS.load_local(faiss_name, embeddings)
    return vectorStores


# def get_conversation_chain(vectorStores: FAISS):
#     llm = ChatOpenAI(deployment_name=...,
#                      model_name=...,
#                      openai_api_base=...,
#                      openai_api_version=...,
#                      openai_api_key=openai.api_key)
#
#     compressor = LLMChainExtractor.from_llm(llm)
#     base_retriever = vectorStores.as_retriever(search_type="mmr", search_kwargs={"k": 10})
#     compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
#
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=compression_retriever,
#         return_source_documents=True,
#         verbose=False,
#         chain_type="stuff"
#     )
#     return conversation_chain
#
#
# def ask_question(qa, question: str, chat_history):
#     query = f""
#
#     result = qa({"question": query, "chat_history": chat_history})
#     print(result)
#     print("Question:", question)
#     print("Answer:", result["answer"])
#
#     print(result)
#
#     return result["answer"]


def generate_response(query: str, history, vectorstore) -> str:
    knowledge = []
    for doc in vectorstore.max_marginal_relevance_search(query, k=10):
        knowledge.append(doc)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "system",
                "content": ("You are a software developer bot trained to assist in coding projects. Your manager has "
                            "provided you with a knowledge base of ICP documents stored in a vector store and general "
                            "programming knowledge. Your role "
                            "is to generate Python code based on these documents and to answer any technical "
                            "questions your manager may have."
                            )
            },
            {
                "role": "system",
                "content": f"Content from Vector Store based on relevance of query: {knowledge}"
            },
            {
                "role": "system",
                "content": f"Previous Question to keep track on Conversation: {history}"
            },
            {
                "role": "user",
                "content": f"Your manager asks:: {query}. "
            }
        ],

        temperature=0,
        max_tokens=5000,
        top_p=0.4,
        frequency_penalty=1.5,
        presence_penalty=1
    )
    bot_response = response['choices'][0]['message']['content']
    return bot_response
