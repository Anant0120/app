import os
import time
import fitz  # PyMuPDF
import openai
import streamlit as st
from pinecone import Pinecone, Index
from openai.error import ServiceUnavailableError
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Initialize Pinecone client
pinecone_client = Pinecone(api_key="YOUR_PINECONE_API_KEY")

# Specify the index name
index_name = "transcript"
index = pinecone_client.Index(index_name)

# Set OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Helper Functions
def pdf_to_text(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=2000):
    """Split text into chunks of a given size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def get_embedding(text, retries=5):
    """Get the embedding of the text using OpenAI with retry mechanism."""
    for attempt in range(retries):
        try:
            response = openai.Embedding.create(
                input=[text],
                model="text-embedding-ada-002"  # Use an appropriate model for embeddings
            )
            return response['data'][0]['embedding']
        except ServiceUnavailableError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise

def query_pinecone(query, top_k=5):
    """Query Pinecone to find the most relevant document chunks."""
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [res['metadata']['text'] for res in results['matches']]

def generate_response(query):
    """Generate a response using GPT-3.5-turbo based on retrieved documents."""
    retrieved_texts = query_pinecone(query)
    combined_text = "\n\n".join(retrieved_texts)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Answer the question based on the following documents:\n\n{combined_text}\n\nQuestion: {query}\nAnswer:"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0]['message']['content'].strip()

def summarize_text(text):
    """Summarize the given text using OpenAI."""
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Summarize the following text:\n\n{text}",
        max_tokens=150,
        temperature=0.5
    )
    return response.choices[0].text.strip()

def save_summary_to_pdf(summary, file_path):
    """Save summary text to a PDF file."""
    c = canvas.Canvas(file_path, pagesize=letter)
    c.drawString(100, 750, "Summary")
    text_object = c.beginText(100, 730)
    text_object.setTextOrigin(100, 730)
    text_object.setFont("Helvetica", 10)
    text_object.textLines(summary)
    c.drawText(text_object)
    c.save()

# Streamlit Interface
st.title("PDF Question Answering System")

# Section 1: Welcome and PDF Upload
st.header("Welcome")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract text from the PDF
    text = pdf_to_text("temp.pdf")
    chunks = chunk_text(text)
    
    # Upsert chunks into Pinecone
    for chunk in chunks:
        embedding = get_embedding(chunk)
        index.upsert([(str(hash(chunk)), embedding, {'text': chunk})])
    
    # Extract basic information
    st.subheader("Document Overview")
    st.write("Company Name: [Extracted Company Name]")
    st.write("Parties Involved: [Extracted Parties]")

    # Section 2: Raw Text and Topics
    st.header("Raw Text and Topics")
    st.subheader("Raw Text")
    st.write(text)

    st.subheader("Topics")
    topics = ["Topic 1", "Topic 2", "Topic 3"]  # Placeholder for topic extraction logic
    selected_topic = st.selectbox("Select a topic to get summary", topics)

    if selected_topic:
        summary = summarize_text(selected_topic)
        st.write(f"Summary of {selected_topic}: {summary}")

        if st.button("Download Summary as PDF"):
            save_summary_to_pdf(summary, "summary.pdf")
            with open("summary.pdf", "rb") as pdf_file:
                st.download_button("Download Summary", data=pdf_file, file_name="summary.pdf", mime="application/pdf")

    if st.button("Get Summary of All Topics"):
        all_summaries = "\n\n".join([summarize_text(topic) for topic in topics])
        st.write("Summary of All Topics:")
        st.write(all_summaries)
        
        save_summary_to_pdf(all_summaries, "all_summaries.pdf")
        with open("all_summaries.pdf", "rb") as pdf_file:
            st.download_button("Download All Summaries", data=pdf_file, file_name="all_summaries.pdf", mime="application/pdf")

    # Section 3: Question Answer Summary
    st.header("Question Answer Summary")
    st.subheader("Raw Text")
    st.write(text)

    st.subheader("Topics")
    selected_topic_qa = st.selectbox("Select a topic to get summary for QA", topics)

    if selected_topic_qa:
        summary_qa = summarize_text(selected_topic_qa)
        st.write(f"Summary of {selected_topic_qa}: {summary_qa}")

        if st.button("Download QA Summary as PDF"):
            save_summary_to_pdf(summary_qa, "qa_summary.pdf")
            with open("qa_summary.pdf", "rb") as pdf_file:
                st.download_button("Download QA Summary", data=pdf_file, file_name="qa_summary.pdf", mime="application/pdf")

    if st.button("Get QA Summary of All Topics"):
        all_qa_summaries = "\n\n".join([summarize_text(topic) for topic in topics])
        st.write("QA Summary of All Topics:")
        st.write(all_qa_summaries)
        
        save_summary_to_pdf(all_qa_summaries, "all_qa_summaries.pdf")
        with open("all_qa_summaries.pdf", "rb") as pdf_file:
            st.download_button("Download All QA Summaries", data=pdf_file, file_name="all_qa_summaries.pdf", mime="application/pdf")

    # Section 4: Chatbot
    st.header("Chatbot")
    st.write("Ask questions related to the uploaded PDF.")
    user_question = st.text_input("Enter your question:")

    if user_question:
        chatbot_response = generate_response(user_question)
        st.write("Answer:")
        st.write(chatbot_response)
