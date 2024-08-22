import os
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Function to read QA pairs from uploaded text file
def read_qa_pairs_from_file(file_path):
    qa_pairs = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            qa_pairs.append(line.strip())
    return qa_pairs

# Function to split text into chunks and store each chunk in a local file
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        chunk_file_path = f"chunks/chunk{i+1}.txt"
        with open(chunk_file_path, "w", encoding="utf-8") as f:
            f.write(chunk)
    return chunks

# Function to create a vector store from text chunks
def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Function to create a conversational chain
async def get_conversational_chain():
    prompt_template = """
    Use the context below to answer the question. Provide as detailed a response as possible. If the answer is not in
    the context, state "answer is not available in the context."

    Context: {context}

    Question: {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to handle user input and generate a response
async def process_user_input(user_question, vector_store):
    docs_and_scores = vector_store.similarity_search_with_score(user_question)
    # Sort documents by their similarity scores in descending order and get top 4
    top_docs_and_scores = sorted(docs_and_scores, key=lambda x: x[1], reverse=True)[:4]
    top_docs, top_scores = zip(*top_docs_and_scores)

    if not top_docs:
        return "No relevant documents found.", None

    results = []
    for i, (doc, score) in enumerate(zip(top_docs, top_scores)):
        results.append({
            "Chunk Number": i + 1,
            "Content": doc.page_content,
            "Similarity Score": score
        })

    # Run conversational chain to get the answer
    chain = await get_conversational_chain()
    response = chain({"input_documents": top_docs, "question": user_question}, return_only_outputs=True)

    return results, response["output_text"]

# Function to process uploaded text file
def process_uploaded_text_file(file_content):
    os.makedirs("chunks", exist_ok=True)
    text_chunks = split_text_into_chunks(file_content)
    if not text_chunks:
        return None, "Failed to split text into chunks."

    vector_store = create_vector_store(text_chunks)
    return vector_store, None

# Streamlit app
st.title('BRSR Report RAG Application')

uploaded_file = st.file_uploader("Upload QA pairs text file", type=["txt"])

if uploaded_file:
    file_content = uploaded_file.read().decode('utf-8')
    vector_store, processing_error = process_uploaded_text_file(file_content)
    
    if processing_error:
        st.error(processing_error)
    else:
        user_question = st.text_input("Enter your question:")

        if user_question:
            user_results, answer = asyncio.run(process_user_input(user_question, vector_store))
            
            # Track from which chunk the answer is retrieved
            retrieved_from_chunk = None
            for result in user_results:
                if answer in result["Content"]:
                    retrieved_from_chunk = result['Chunk Number']
                    break

            st.write(f"Question: {user_question}")
            st.write(f"Answer: {answer}")

            if retrieved_from_chunk:
                st.write(f"Answer retrieved from Chunk: {retrieved_from_chunk}")

            # Display chunk information
            st.subheader("Chunks Information:")
            for result in user_results:
                st.write(f"Chunk Number: {result['Chunk Number']}")
                st.write(f"Similarity Score: {result['Similarity Score']}")
                st.write(f"Content:")
                st.code(result["Content"])
