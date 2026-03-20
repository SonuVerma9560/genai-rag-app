import streamlit as st
import openai

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# Set API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("📄 Chat with PDF (GenAI RAG App)")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded!")

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

    # Store in FAISS
    db = FAISS.from_documents(docs, embeddings)

    query = st.text_input("Ask a question from the PDF:")

    if query:
        results = db.similarity_search(query)

        context = "\n".join([doc.page_content for doc in results])

        # OpenAI call
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer based only on the given context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )

        st.write(response["choices"][0]["message"]["content"])