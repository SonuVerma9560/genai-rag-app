import streamlit as st
import openai
from pypdf import PdfReader

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("📄 Chat with PDF")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)

    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    st.success("PDF loaded!")

    question = st.text_input("Ask question:")

    if question:
        prompt = f"""
        Answer based on this document:

        {text[:2000]}

        Question: {question}
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        st.write(response["choices"][0]["message"]["content"])