
import os

from utils import *

path  = './data/Understanding_Climate_Change.pdf'


def encode_pdf(path, chunk_size = 1000, chunk_overlap = 200):
    try:
        
        # load pdf
        pdf_loader = PyPDFLoader(path)
        documents = pdf_loader.load()

        # split documents into chunk
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
        )
        texts = text_splitter.split_documents(documents)
        cleaned_text = replace_t_with_space(texts)

        ollama_emb = OllamaEmbeddings(
                model='nomic-embed-text', 
                show_progress=True  
            )
        vectorstore = FAISS.from_documents(
            documents=cleaned_text,
            embedding=ollama_emb
        )  
        return vectorstore
    except Exception as ex:
        print(f'error is coming from given encode_documents :: {ex}')



