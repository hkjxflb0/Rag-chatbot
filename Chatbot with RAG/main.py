from typing import List

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
import ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.schema import Document
import io
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

app = FastAPI()

# Mount the static directory for serving CSS and JS files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2Templates with the templates directory
templates = Jinja2Templates(directory="templates")

# Initialize FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docstore = InMemoryDocstore({})
documents = []  # No documents at the start

# Check if there are documents before initializing FAISS
if documents:
    faiss_index = FAISS.from_documents(documents, embeddings, docstore=docstore)
    print("Succesfull")
else:
    faiss_index = None  # Or handle this differently based on your needs
    print("No documents to index. FAISS index not initialized.")


# Data model for chat requests
class ChatRequest(BaseModel):
    query: str
class Rag_Words:
    rag_keywords = []

# Utility function to determine if RAG is required for the query
def requires_rag(query: str) -> bool:
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.split()
        text = [word for word in text if word not in stop_words]
        return text


    query = preprocess_text(query)
    query = ' '.join(query)
    query = nltk.word_tokenize(query)
    print(query,"*")
    print(Rag_Words.rag_keywords,"++")
    intersection = [value for value in query if value in Rag_Words.rag_keywords]
    print(len(intersection))
    if len(intersection) > 1:
        print("True")
        return  True
    return False



@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    # Render the index.html template
    return templates.TemplateResponse("index.html", {"request": request})


import os
from tempfile import NamedTemporaryFile
from langchain.document_loaders import PyPDFLoader

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Wrap file content in io.BytesIO for PDF loader
    content = await file.read()
    file_stream = io.BytesIO(content)
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(content)

        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path )

    # Load documents from PDF and convert them
    pdf_documents = loader.load()

    new_documents = [Document(page_content=doc.page_content, metadata={"source": file.filename}) for doc in pdf_documents]
    print(new_documents)



    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.split()
        text = [word for word in text if word not in stop_words]
        return text


    words = preprocess_text(new_documents[0].page_content)
    preprocessed_string = ' '.join(words)
    words = nltk.word_tokenize(preprocessed_string)

    Rag_Words.rag_keywords = words
    print(Rag_Words.rag_keywords)
    global faiss_index  # To ensure we're updating the global variable
    if faiss_index is None:
        # Initialize FAISS if it hasn't been initialized yet
        faiss_index = FAISS.from_documents(new_documents, embeddings, docstore=docstore)
    else:
        # Add new documents to the existing FAISS index
        faiss_index.add_documents(new_documents)

    return {"message": "Document indexed successfully"}


@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.query
    if requires_rag(query):
        if faiss_index is not None:
            # Perform similarity search if FAISS index is initialized
            results = faiss_index.similarity_search(query, k=3)
            context = results[0].page_content if results else ""
            context = f"'{context}'+In short, to the point"
        else:
            # Return a fallback message if no documents are indexed
            context = "No documents available to search. Please upload a document."
        summary = f"'{query}'+In short, to the point"
        response = ollama.generate(model="llama3.2:1b", prompt=summary+ "\nContext: " + context)
    else:
        context =f"'{query}'+In short, to the point"
        response = ollama.generate(model="llama3.2:1b", prompt=context)

    return {"response": response['response']}
