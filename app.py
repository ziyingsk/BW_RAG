import os
import streamlit as st
from dotenv import load_dotenv
import itertools
from pinecone import Pinecone
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set up environment variables and Pinecone database
load_dotenv()  # Load environment variables from a .env file
cache_dir = os.getenv("CACHE_DIR")  # Directory for cache
Huggingface_token = os.getenv("API_TOKEN")  # Huggingface API token
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))  # Pinecone API key
index = pc.Index(os.getenv("Index_Name"))  # Pinecone index name

# Initialize embedding model (LLM will be saved to cache_dir if assigned)
embedding_model = "all-mpnet-base-v2"  # Pretrained model name for embeddings

if cache_dir:
    embedding = SentenceTransformer(embedding_model, cache_folder=cache_dir)
else:
    embedding = SentenceTransformer(embedding_model)

# Function to read PDF files and return documents
def read_doc(file_path):
    file_loader = PyPDFDirectoryLoader(file_path)
    documents = file_loader.load()
    return documents

# Function to split documents into chunks
def chunk_data(docs, chunk_size=300, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return doc

# Function to break an iterable into chunks of specified size
def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

# Start of Streamlit interface
st.title("RAG-Anwendung (RAG Application)")
st.caption("Diese Anwendung kann Ihnen helfen, kostenlos Fragen zu PDF-Dateien zu stellen. (This application can help you ask questions about PDF files for free.)")

# Upload a PDF file
uploaded_file = st.file_uploader("Wählen Sie eine PDF-Datei, das Laden kann eine Weile dauern. (Choose a PDF file, loading might take a while.)", type="pdf")
if uploaded_file is not None:
    # Ensure the temp directory exists and is empty
    temp_dir = "tempDir"
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)  # Only removes empty directories

    os.makedirs(temp_dir, exist_ok=True)

    # Save the uploaded file temporarily
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process the uploaded PDF file
    doc = read_doc(temp_dir + "/")
    documents = chunk_data(docs=doc)
    texts = [document.page_content for document in documents]
    pdf_vectors = embedding.encode(texts)
    vector_count = len(documents)
    example_data_generator = map(lambda i: (f'id-{i}', pdf_vectors[i], {"text": texts[i]}), range(vector_count))
    
    # Update the Pinecone index with new vectors
    if 'ns1' in index.describe_index_stats()['namespaces']:
        index.delete(delete_all=True, namespace='ns1')
    for ids_vectors_chunk in chunks(example_data_generator, batch_size=100):
        index.upsert(vectors=ids_vectors_chunk, namespace='ns1')

# Input for the search query
sample_query = st.text_input("Stellen Sie eine Frage zu dem PDF: (Ask a question related to the PDF:)")
if st.button("Abschicken (Submit)"):
    if uploaded_file is not None and sample_query:
        # Encode the query and search in the Pinecone index
        query_vector = embedding.encode(sample_query).tolist()
        query_search = index.query(vector=query_vector, top_k=5, include_metadata=True，namespace='ns1')

        matched_contents = [match["metadata"]["text"] for match in query_search["matches"]]

        # Rerank the matched contents
        rerank_model = "BAAI/bge-reranker-v2-m3"
        if cache_dir:
            tokenizer = AutoTokenizer.from_pretrained(rerank_model, cache_dir=cache_dir)
            model = AutoModelForSequenceClassification.from_pretrained(rerank_model, cache_dir=cache_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(rerank_model)
            model = AutoModelForSequenceClassification.from_pretrained(rerank_model)
        model.eval()

        pairs = [[sample_query, content] for content in matched_contents]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=300)
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
            matched_contents = [content for _, content in sorted(zip(scores, matched_contents), key=lambda x: x[0], reverse=True)]
            matched_contents = matched_contents[0]
        del model
        torch.cuda.empty_cache()

        # Display matched contents after reranking
        st.markdown("### Möglicherweise relevante Abschnitte aus dem PDF (Potentially relevant sections from the PDF):")
        st.write(matched_contents)

        # Generate an answer using a language model
        query_model = "meta-llama/Meta-Llama-3-8B-Instruct"
        llm_huggingface = HuggingFaceHub(repo_id=query_model, model_kwargs={"temperature": 0.7, "max_length": 500})

        prompt_template = PromptTemplate(input_variables=['query', 'context'], template="{query}, Beim Beantworten der Frage bitte mit dem Wort 'Antwort:' beginnen，unter Berücksichtigung des folgenden Kontexts: \n\n{context}")

        prompt = prompt_template.format(query=sample_query, context=matched_contents)
        chain = LLMChain(llm=llm_huggingface, prompt=prompt_template)
        result = chain.run(query=sample_query, context=matched_contents)

        # Polish the answer
        result = result.replace(prompt, "")
        special_start = "Antwort:"
        start_index = result.find(special_start)
        if start_index != -1:
            result = result[start_index + len(special_start):].lstrip()
        else:
            result = result.lstrip()

        # Display the final answer with a note about limitations
        st.markdown("### Antwort (Answer):")
        st.write(result)
        st.markdown("**Hinweis:** Aufgrund begrenzter Rechenleistung kann das große Sprachmodell möglicherweise keine vollständige Antwort liefern. (Note: Due to limited computational power, the large language model might not be able to provide a complete response.)")
