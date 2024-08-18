
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama 
from langchain_community.vectorstores import FAISS
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_community.embeddings import OllamaEmbeddings

from langchain.prompts import PromptTemplate
from typing import List
import numpy as np
import random
import textwrap



def replace_t_with_space(list_of_documents):
    try:
        """
        Replace all the tab characters ('\t') with space in page contained of each documents:

        Args:
            list_of_documents: A list of documents objects, each with 'page_content' attribute.

        return:
            the modified list of documents replace all tab with space.
        """

        for doc in list_of_documents:
            doc.page_content = doc.page_content.replace('\t', ' ')
        return list_of_documents
    
    except Exception as e:
        print(f'error is raised from replace_t_with_space :: {e}')


def text_wrap(text, width = 120):
    try:
        """
        Wrap the text to specified width.

        Args:
            text: Input text to wrap.
            width: The width of the wrapped text.

        return:
        The wrapped text.
        """
        return textwrap.fill(text, width=width)
    except Exception as e:
        print(f'error is raised from text wrap function :: {e}')


def encode_pdf(path, chunk_size =1000, chunk_overlap = 200):
    try:
        """
        Encode the pdf documents into the vector store with the help of open-source embedding model/ open-ai emb model.

        Args:
            path: The path to the pdf document.
            chunk_size: desired size of each text size.
            chunk_overlap: amount of overlap between each chunk.

        return:
            A FAISS vector store containing the encode of documents.
        """

        # load the documents
        pdf_loader = PyPDFLoader(path)
        documents = pdf_loader.load()

        # split the documents into chunk:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function= len
            
        )

        texts = text_splitter.split_documents(text_splitter)
        cleaned_texts = replace_t_with_space(texts)

        # Store into vector storage.
        ollama_emb = OllamaEmbeddings(
                model='nomic-embed-text', 
                show_progress=True  
            )
        vectorestore = FAISS.from_documents(
            documents=cleaned_texts,
            embedding=ollama_emb
        )
        return vectorestore


    except Exception as e:
        print(f'error is raised from encode pdf :: {e}')









