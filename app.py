# Part of the code was from the repo: https://github.com/krishnaik06/Complete-Langchain-Tutorials
"""
process:
 1. Read and divided pdf into chucks
 2. Embedde chucks and save to database
 3. Search query context in the database
 4. Rerank the contexts
 5. Answer = Query + context + LLM
"""
import os
from dotenv import load_dotenv
import itertools
from pinecone import Pinecone
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
#from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForQuestionAnswering
#import pandas as pd
from sentence_transformers import SentenceTransformer

"""
set up enviroment, Pinecone is a datebase
"""

load_dotenv() # load document .env
cache_dir = os.getenv("CACHE_DIR") # dir for cache
Huggingface_token = os.getenv("API_TOKEN") #Huggingface_api_key
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY")) #database_api_key
index = pc.Index(os.getenv("Index_Name")) #database index_name
database_dataready = os.getenv("DATA_READY") #if data already stored in database

"""
initialize embedding model(llm will be saved to cache_dir if assigned)
"""
embedding_model="all-mpnet-base-v2" #see link https://www.sbert.net/docs/pretrained_models.html

if cache_dir:
    embedding = SentenceTransformer(embedding_model,cache_folder=cache_dir)
else:
    embedding = SentenceTransformer(embedding_model)

"""
Read the pdf files, divie them into chunks and Embedding
"""
def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents

def chunk_data(docs,chunk_size=300,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return doc

if database_dataready is False:
    doc=read_doc('pdfs/')
    print("Number of pages:",len(doc))
    documents=chunk_data(docs=doc)
    print("Number of chunks:",len(documents))
    texts = [document.page_content for document in documents]
    pdf_vectors = embedding.encode(texts)
    print("Number of vectors:", len(pdf_vectors))
    print("Number of dimensions:",len(pdf_vectors[0]))

"""
Save embeddings to database
"""

def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

if database_dataready is False:
     vector_count = len(documents)
     example_data_generator = map(lambda i: (f'id-{i}', pdf_vectors[i],{"text": texts[i]}), range(vector_count))
     for ids_vectors_chunk in chunks(example_data_generator, batch_size=100):
         index.upsert(vectors=ids_vectors_chunk) 

"""
Search query related context
"""
sample_query = "Welche Fähigkeiten sollen Schüler und Schülerinnen im Bereich 'monologisch sprechen' haben?"
query_vector = embedding.encode(sample_query).tolist()
query_search = index.query(
    #namespace="(default)",
    vector=query_vector,
    top_k=5,
    include_metadata=True
)

matched_contents = [match["metadata"]["text"] for match in query_search["matches"]]

"""
Rerank
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

#rerank_context_num = 2 #number of contexts after rerank
rerank_model = "BAAI/bge-reranker-v2-m3"
if cache_dir:
    tokenizer = AutoTokenizer.from_pretrained(rerank_model,cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(rerank_model,cache_dir=cache_dir)
else:
    tokenizer = AutoTokenizer.from_pretrained(rerank_model)
    model = AutoModelForSequenceClassification.from_pretrained(rerank_model)
model.eval()

pairs = [[sample_query, content] for content in matched_contents]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=300)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)
matched_contents = [content for _, content in sorted(zip(scores, matched_contents), key=lambda x: x[0], reverse=True)]
matched_contents = matched_contents[:2]
matched_contents = "\n\n".join(matched_contents)
del model
torch.cuda.empty_cache()

"""
Get answer
"""
query_model="meta-llama/Meta-Llama-3-8B-Instruct" 
llm_huggingface=HuggingFaceHub(repo_id=query_model,
                               model_kwargs={"temperature":0.7,"max_length":500}) #API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"

prompt_template=PromptTemplate(input_variables=['query','context'],
                               template="{query}, Beim Beantworten der Frage bitte mit dem Wort 'Antwort' beginnen，unter Berücksichtigung des folgenden Kontexts: \n\n{context}, bitte ")

#prompt_template.format(query=sample_query, context=matched_contents)
chain=LLMChain(llm=llm_huggingface,prompt=prompt_template)
print(chain.run(query=sample_query, context=matched_contents))