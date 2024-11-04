import fitz
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from llamaapi import LlamaAPI
import json
from transformers import AutoTokenizer #to use Huggingface model
import numpy as np

def extract_text_with_spaces(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def get_relevant_segment(chunks_vectors,query_embed,chunks):
    def get_similarity(embeds,query_embed):
        A = np.array([embeds])
        B = np.array([query_embed])

        cosine_similarity_result = cosine_similarity(A, B)
        return cosine_similarity_result

    max_similarity=-1
    relevant_segment=""
    for i in range(len(chunks_vectors)):
        cosine_similarity_result=get_similarity(chunks_vectors[i],query_embed)
        if(max_similarity<cosine_similarity_result):
            max_similarity=cosine_similarity_result
            relevant_segment=chunks[i]

        # print(f"Cosine Similarity using scikit-learn: {cosine_similarity_result[0][0]}")

    # print(relevant_segment)
    return(relevant_segment)

def ask_llama(llama,content,query):
    api_request_json = {
    "model": "llama3.1-70b",
    "messages": [
        {"role": "system", "content": "Using the content provided, answer the following question. If you cannot locate a precise answer within the content, respond with 'No answer found.' Ensure the answer is direct and factual, avoiding any fabricated or speculative details. If a concise, direct answer is possible, provide it in that format. Do not hallucinate."},
        {"role": "user", "content": content},
        {"role": "user", "content": query},
    ]
    }


    # Make your request and handle the response
    response = llama.run(api_request_json)
    answer=response.json()["choices"][0]["message"]["content"]
    return(answer)

def main(question,file_path):
    llama_api=""
    open_ai_api=""
    api_token = llama_api
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=open_ai_api
    )


    text = extract_text_with_spaces(file_path)
    # with open("content.txt","w",encoding="utf-8") as f:
    #     f.write(text.replace("\n"," "))
    # Tokenize the text using Hugging face model
    tokens = tokenizer.tokenize(text)
    num_tokens = len(tokens)
    text_splitter = SemanticChunker(OpenAIEmbeddings(api_key=open_ai_api), breakpoint_threshold_type="percentile", breakpoint_threshold_amount=100)

    docs = text_splitter.create_documents([text])
    chunks=[]
    for doc in docs:
        chunks.append(doc.page_content)


    chunks_vectors=embeddings.embed_documents(chunks)

    query_embed=embeddings.embed_query(question)
    relevant_segment=get_relevant_segment(chunks_vectors,query_embed,chunks)
    # print(relevant_segment)

    llama = LlamaAPI(api_token)
    final_response=ask_llama(llama,relevant_segment,question)
    return final_response


# file_path="test.pdf"
# main("what is contact number and email id",file_path)