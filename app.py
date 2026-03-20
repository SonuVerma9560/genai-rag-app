import streamlit as st
import openai
from pypdf import PdfReader

# Set API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("📄 Chat with PDF (Simple RAG App)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)

    text = ""
    for page in reader.pages:
        text += page.extract_text()

    st.success("PDF loaded!")

    query = st.text_input("Ask a question:")

    if query:
        prompt = f"""
        Answer the question based only on the context below.

        Context:
        {text[:3000]}

        Question:
        {query}
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        st.write(response["choices"][0]["message"]["content"])