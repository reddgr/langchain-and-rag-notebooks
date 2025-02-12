import streamlit as st  
from langchain_community.document_loaders import PDFPlumberLoader  

# Streamlit file uploader  
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")  

if uploaded_file:  
    # Save PDF temporarily  
    with open("temp.pdf", "wb") as f:  
        f.write(uploaded_file.getvalue())  

    # Load PDF text  
    loader = PDFPlumberLoader("temp.pdf")  
    docs = loader.load()

# Split text into semantic chunks  
text_splitter = SemanticChunker(HuggingFaceEmbeddings())  
documents = text_splitter.split_documents(docs)