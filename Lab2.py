import streamlit as st
from openai import OpenAI, AuthenticationError
from pypdf import PdfReader
#Store API Key
open_api_key = st.secrets["OPEN_API_KEY"]
client = OpenAI(api_key=open_api_key)
#Step 3 :To recieve the PDF uploaded 
#To review each page of the uploaded pdf
def read_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    document = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            document += page_text
    return document

# Show title and description.
st.title("My Document Summarizer")
st.write(
    "Upload a document below and choose how you want your document summarized."
)
    # Create an OpenAI client.
    # To check if the OpenAI key is correct 
try:
        client.models.list()
except AuthenticationError: #If OpenAI rejects the incorrect key
        st.error("Incorrect OpenAI API key. Please check your key and try again.")
        st.stop() 

    # Let the user upload a file via `st.file_uploader`.
uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)", type=("txt", "pdf")
    )

    # User summarizer options 
summary_type = st.sidebar.radio(
        ':red[ Choose a summary type:]',
        [
            "100 words",
            "Two paragraphs",
            "Five bullet points"
        ]
    )
#Summary selection 
if summary_type == "100 words":
        summary_instruction = "Summarize this document in 100 words."
elif summary_type == "Two paragraphs":
        summary_instruction = "Summarize this document in 2 connecting paragraphs."
else:
        summary_instruction = "Summarize this document in 5 bullet points."
# Model selection
advanced_model = st.sidebar.checkbox ("Click for advanced model")
#OpenAI models 
if advanced_model:
        model_to_use = "gpt-5.6-terra"
else:
      model_to_use = "gpt-5.6-luna"


if uploaded_file and summary_type:
        file_extension = uploaded_file.name.split('.')[-1]
        if file_extension == 'txt':
            document = uploaded_file.read().decode()
        elif file_extension == 'pdf':
            document = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")
if uploaded_file:
# Changed question to summary instruction to output summary
    messages = [
                {
                    "role": "user",
                    "content": f"Here's a document: {document} \n\n---\n\n {summary_instruction}",
                }
            ]
    

            # Generate an answer using the OpenAI API.
            #For user to choose which model
    stream = client.chat.completions.create(
                model= model_to_use,
                messages=messages,
                stream=True,
            )

            # Stream the response to the app using `st.write_stream`.
    st.write_stream(stream)