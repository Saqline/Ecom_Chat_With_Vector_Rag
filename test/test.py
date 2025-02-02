import os
import uuid
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



####--Custom Json----###
model_name="llama3.2-vision:11b-instruct-q4_K_M"
# model_name="llama3.2:1b-instruct-q5_0"  


jq_schema = ".[] | {content: .expanded_description, metadata: {id: .ID, product_name: .\"Product Name\", current_price: .\"Current Price\", original_price: .\"Original Price\", seller_name: .\"Seller Name\"}}"

file_path = "modified_data.json"

with open(file_path, "r", encoding="utf-8") as f:
    json_content = f.read()

with open(file_path, "w", encoding="utf-8") as f:
    f.write(json_content)

loader = JSONLoader(file_path=file_path, jq_schema=jq_schema, text_content=False)
documents = loader.load()



text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(documents)

texts = [doc.page_content for doc in split_documents]
metadatas = [{"id": str(uuid.uuid4()), **doc.metadata} for doc in split_documents]

faiss_index_path = "faiss_index"
embeddings = OllamaEmbeddings(model=model_name)

if not os.path.exists(faiss_index_path):
    print("Creating new FAISS index...")
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vectorstore.save_local(faiss_index_path)
else:
    print("Loading existing FAISS index...")
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True) 
    


query = "Current Price between 100 to 500 of Sharee Petticoat list details"
retrieved_docs = vectorstore.similarity_search(query, k=25)

# retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
# retrieved_docs = retriever.invoke(query)

if retrieved_docs:
    retrieved_content = "\n\n".join(
        [f"- {doc.page_content}" for doc in retrieved_docs]
    )
    prompt = (
        f"Based on the following information: \n\n{retrieved_content}\n\n"
        f"Answer this  question's best match:\n\n{query}"
    )
    llm = OllamaLLM(model=model_name)
    response = llm.invoke(prompt)

    print("\nLLM Response:", response)
else:
    print("No relevant documents found.")









# ## -- Custom Text Doc Load -- ##

# embeddings = OllamaEmbeddings(model="llama3.2:1b-instruct-q5_0")

# loader = JSONLoader("ai_large_document.txt")
# documents = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# split_documents = text_splitter.split_documents(documents)

# vectorstore = InMemoryVectorStore.from_documents(
#     split_documents, embedding=embeddings
# )

# retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# retrieved_documents = retriever.invoke("Tell me about Generative AI research and its applications.")
# print("Retrieved Documents:", retrieved_documents)


# llm = OllamaLLM(model="llama3.2:1b-instruct-q5_0")

# if retrieved_documents:
#     prompt = f"Based on the following information: \n\n {retrieved_documents},  \n\n Answer this \n\n  Question: Tell me about Generative AI research and its applications."

#     response = llm.invoke(prompt)

#     print("\nLLM Response:", response)
# else:
#     print("No documents retrieved.")










#     ###--Custom-Text--## 
# embeddings = OllamaEmbeddings(model="llama3.2:1b-instruct-q5_0")
# text = "LangChain is the framework for building context-aware reasoning applications. Bangladesh is a nice country. India is a neighboring country of Bangladesh."
# text2 = (
#     "TechKnowGram Limited is a leading technology company based in Bangladesh "
#     "that specializes in providing web-based customized solutions in various areas, "
#     "including ERP, AI, and Data Analysis."
# )
# vectorstore = InMemoryVectorStore.from_texts(
#     [text, text2],
#     embedding=embeddings,
# )

# retriever = vectorstore.as_retriever()

# llm = OllamaLLM(model="llama3.2:1b-instruct-q5_0")

# qa_system_prompt = (
#     "You are a helpful assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer the question. "
#     "If the context is insufficient, say 'I don't know.' Be concise in your response."
#     "\n\n{context}"
# )
# qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", qa_system_prompt),
#         ("human", "{input}"),
#     ]
# )
# qa_chain = create_stuff_documents_chain(llm, qa_prompt)
# retrieval_chain = create_retrieval_chain(retriever, qa_chain)

# history_aware_retriever = create_history_aware_retriever(
#     llm,
#     retriever,
#     qa_prompt,
# )


# query = "can you tell me about TechKnowGram Limited?"
# result = retrieval_chain.invoke({"input": query})
# print("RAG Response:", result['answer'])


# conversation_history = []

# def interact(query):
#     global conversation_history
#     conversation_history.append(("human", query))

#     response = history_aware_retriever.invoke({"input": query, "history": conversation_history})

    
#     # print("Response Structure:", response)

#     context = "\n".join([doc.page_content for doc in response])
#     prompt = f"Based on the following information: \n\n {context}, \n\n Answer this \n\n  Question: {query}"

#     answer = llm.invoke(prompt)

#     conversation_history.append(("assistant", answer))

#     return answer

# print("User: Can you tell me about TechKnowGram Limited?")
# print("Assistant:", interact("Can you tell me about TechKnowGram Limited?"))

# print("User: What other services do they offer?")
# print("Assistant:", interact("What other services do they offer?"))

# print("User: Where is TechKnowGram Limited based?")
# print("Assistant:", interact("Where is TechKnowGram Limited based?"))







###--Custom Doc Class--###

# class Document:
#     def __init__(self, title, content, author):
#         self.id = str(uuid.uuid4())  
#         self.page_content = content 
#         self.metadata = {
#             # "title": title,
#             # "author": author
#         }

# # Create document instances
# doc1 = Document(
#     title="Meeting Schedule",
#     content="The meeting is scheduled for 2 PM tomorrow.",
#     author="John Doe"
# )

# doc2 = Document(
#     title="Meeting Agenda",
#     content="""
#     # Meeting Agenda

#     ## Date: October 15, 2023
#     ## Time: 2 PM
#     ## Location: Conference Room A

#     ### Agenda Items:
#     1. **Introduction**
#        - Welcome remarks by the CEO
#        - Brief overview of the meeting objectives

#     2. **Project Updates**
#        - Status report on Project Alpha
#        - Progress on Project Beta

#     3. **Financial Review**
#        - Quarterly financial report
#        - Budget allocations for the next quarter

#     4. **Marketing Strategy**
#        - New marketing campaigns
#        - Social media engagement metrics

#     5. **Q&A Session**
#        - Open floor for questions and discussions

#     ### Action Items:
#     - Assign responsibilities for the next quarter's projects
#     - Schedule follow-up meetings for each department

#     ### Closing Remarks:
#     - Summary of the meeting
#     - Next steps and future plans

#     ### Attendees:
#     - John Doe, CEO
#     - Jane Smith, CFO
#     - Alice Johnson, Marketing Director
#     - Bob Brown, Project Manager
#     """,
#     author="Jane Smith"
# )
# from langchain_core.vectorstores import InMemoryVectorStore

# vectorstore = InMemoryVectorStore.from_documents(
#     [doc1, doc2],
#     embedding=embeddings,
# )

# embeddings = OllamaEmbeddings(model="llama3.2:1b-instruct-q5_0")



# # Use the vectorstore as a retriever
# retriever =vectorstore.as_retriever()

# # Retrieve the most similar text
# retrieved_documents = retriever.invoke("About John Doe?")

# # # show the retrieved document's content

# print(retrieved_documents)

# llm = OllamaLLM(model="llama3.2:1b-instruct-q5_0") 


# # Combine retrieved documents into a prompt for the LLM
# if retrieved_documents:
#     prompt = f"Based on the following information: \n\n {retrieved_documents},  \n\n Answer this \n\n  Question: About John Doe?"

#     # Generate a response from the LLM
#     response = llm.invoke(prompt)

#     # Print the response from the LLM
#     print("LLM Response:", response)
# else:
#     print("No documents retrieved.")