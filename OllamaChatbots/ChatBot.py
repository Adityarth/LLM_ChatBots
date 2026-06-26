from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load your PDF file
loader = PyPDFLoader("your_document.pdf")
docs = loader.load()

# 2. Split the PDF text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# 3. Create local embeddings using Ollama
# Nomic-embed-text matches perfectly with local workflows
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. Store the chunks locally in Chroma DB
vector_store = Chroma.from_documents(chunks, embeddings)

# 5. Set up the local database as a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 6. Initialize your local Llama 3.2:3b model
llm = ChatOllama(model="llama3.2:3b", temperature=0)

# 7. Define the system prompt instructions
system_prompt = (
    "You are a helpful assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Keep the answer concise.\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 8. Assemble the Q&A Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 9. Ask your local model a question about your PDF
response = rag_chain.invoke({"input": "What is the main summary of this document?"})

print("Answer:", response["answer"])
