import streamlit as st
from openai import OpenAI
import sys

__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from pathlib import Path
from PyPDF2 import PdfReader

#Page Title
st.title("Lab 4 - RAG Pipeline with Vector DB")

#Open AI Client
if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(
        api_key=st.secrets["OPEN_API_KEY"]
    )


#To extract text from a PDF file
def extract_text_from_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text

#Add text to the ChromaDB collection
def add_to_collection(collection, text, file_name):

    client = st.session_state.openai_client

    #Create embedding
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    embedding = response.data[0].embedding

    #Store document in ChromaDB
    collection.add(
        ids=[file_name],
        documents=[text],
        metadatas=[
            {"file_name": file_name}
        ],
        embeddings=[embedding]
    )
#Load PDFs from a folder into the ChromaDB collection
def load_pdfs_to_collection(folder_path, collection):

    pdf_folder = Path(folder_path)

    pdf_files = list(
        pdf_folder.glob("*.pdf")
    )

    for pdf_file in pdf_files:

        text = extract_text_from_pdf(
            pdf_file
        )

        file_name = pdf_file.name

        add_to_collection(
            collection,
            text,
            file_name
        )

    return len(pdf_files)

#Create vector database 
def create_vector_db():

    #create/open ChromaDB client
    chroma_client = chromadb.PersistentClient(
        path="./ChromaDB_for_Lab4"
    )

    collection = chroma_client.get_or_create_collection(
        name="Lab4Collection"
    )

    #Only create embeddings if the collection is empty
    if collection.count() == 0:

        load_pdfs_to_collection(
            "Lab4-Data",
            collection
        )

    return collection


#Store vector

if "Lab4_VectorDB" not in st.session_state:

    st.session_state.Lab4_VectorDB = (
        create_vector_db()
    )


#To search the vector database
def search_vector_db(collection, search_text):

    client = st.session_state.openai_client

    # Convert the user's search into an embedding
    response = client.embeddings.create(
        input=search_text,
        model="text-embedding-3-small"
    )

    query_embedding = (
        response.data[0].embedding
    )

    # Find the 3 closest documents
    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=3
    )

    return results

#Part B: Course chatbot display
if "Lab4_messages" not in st.session_state:

    st.session_state.Lab4_messages = []


#Display previous messages
for message in st.session_state.Lab4_messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


#Get question from user
prompt = st.chat_input(
    "Ask a question about the courses..."
)


if prompt:

    #Display/save user message
    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.Lab4_messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

#Info from the vector database
    results = search_vector_db(
        st.session_state.Lab4_VectorDB,
        prompt
    )

    retrieved_documents = (
        results["documents"][0]
    )

    retrieved_files = (
        results["ids"][0]
    )


    #Combine retrieved documents
    rag_context = ""

    for i in range(
        len(retrieved_documents)
    ):

        rag_context += (
            f"\n\nSource: "
            f"{retrieved_files[i]}\n"
            f"{retrieved_documents[i]}"
        )

#System prompt for the RAG pipeline
    system_prompt = f"""
You are a helpful course information assistant.

Use the retrieved course information below
to answer the user's question.

If you use information from the retrieved
documents, make it clear by saying:
"Based on the retrieved course information..."

If the retrieved information does not contain
the answer, clearly say that the answer was
not found in the retrieved course information.

Retrieved course information:

{rag_context}
"""


    #Create messages sent to OpenAI
    messages_for_llm = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    messages_for_llm.extend(
        st.session_state.Lab4_messages
    )
#Send messages to OpenAI and get response
    client = st.session_state.openai_client

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages_for_llm
    )

    assistant_response = (
        response.choices[0].message.content
    )


    #Display assistant response
    with st.chat_message("assistant"):

        st.write(
            assistant_response
        )

        st.caption(
            "RAG sources: "
            + ", ".join(retrieved_files)
        )


    #Save assistant response
    st.session_state.Lab4_messages.append(
        {
            "role": "assistant",
            "content": assistant_response
        }
    )