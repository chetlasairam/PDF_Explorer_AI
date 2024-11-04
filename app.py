import streamlit as st
import os
import fitz
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from backend import *


# Streamlit app
st.title("PDF Explorer")

# Sidebar inputs
folder_path = st.sidebar.text_input("Enter the path of your local folder:")
question = st.sidebar.text_input("Enter your question:")

def get_llm_responces(pdf_path,question):
    try:
        doc = fitz.open(pdf_path)
        # text = "".join(page.get_text() for page in doc)
        doc.close()
        return {
            'Filename': os.path.basename(pdf_path),
            'Text Length': main(question, pdf_path)
        }
    except Exception as e:
        return {
            'Filename': os.path.basename(pdf_path),
            'Text Length': 0
        }

if folder_path and os.path.exists(folder_path):
    # Get all PDF paths
    pdf_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files if file.lower().endswith('.pdf')
    ]
    
    if pdf_paths:
        progress_bar = st.progress(0)
        status_text = st.empty()
        # Prepare arguments for each PDF
        args_list = [(pdf_path, question) for pdf_path in pdf_paths]
        # Process PDFs using ThreadPool
        results = []
        with ThreadPoolExecutor() as executor:
            futures = executor.map(lambda args: get_llm_responces(*args), args_list)
            for i, result in enumerate(futures):
                results.append(result)
                progress_bar.progress((i + 1) / len(pdf_paths))
                status_text.text(f"Processing {i + 1} of {len(pdf_paths)} PDFs")
            
        # Display results
        df = pd.DataFrame(results)
        st.dataframe(df[['Filename', 'Text Length']], hide_index=True)
    else:
        st.info("No PDF files found in the specified folder.")
else:
    st.error("Please enter a valid folder path")