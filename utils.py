
from langchain_community.document_loaders import PyPDFLoader
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
import fitz

from langchain.docstore.document import Document


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



def retrieve_context_per_question(question, chunk_query_retriever):
    try:
        """
        Retrivers relevant context and unique URLs from given question using the chunk query retriever.
        
        Args:
            question : the question for which to retrieve contex and URLs.
        
        returns:
            A tuple containing:
            - string with the concatenated content of relevent documents.
            - list of unique URLs from metadata of the relevant documents.
        """
        
        docs = chunk_query_retriever.get_relevant_documents(question)

        context = [doc.page_content for doc in docs]

        return context
    
    except Exception as e:
        print(f'error is coming from given retrieve context per question :: {e}')


def show_context(context):
    try:
        
        for i, c in enumerate(context):
            print(f"Context {i+1}:")
            print(c)
            print("\n")
    except Exception as e:
        print(f'error is raised from show contexc function :: {e}')



def create_question_answer_from_context_chain(llm):
    try:
        


        question_answer_context_llm = llm

        question_answer_prompt_template = """

        For the question below, provide a concise but suffice answer based only on the provided 
        context:{context}
        Question:{question}

        """
        question_answer_from_context_prompt= PromptTemplate(
            template=question_answer_prompt_template,
            input_variables=["context", 'question']
        )

        # create a chain by combining the prompt template and the llm model
        question_answer_from_context_chain = question_answer_from_context_prompt | question_answer_context_llm
        return question_answer_from_context_chain
    
    except Exception as e:
        print(f'error is raised from create question answer from context chain :: {e}')


def answer_question_from_context(question, context, question_answer_from_context_chain):
    
    try:
        """
        Answer a question using the given context by invoking a chain reasoning.
        
        Args:
            question (str): the question to be answered.
            context (str): the context to be used for answering the question.
        return:
            a dictionary the answer, context and question.
        """
        input_data = {
            'question': question,
            'context': context
        }
        output = question_answer_from_context_chain.invoke(input_data)
        answer = output.answer_based_on_content
        return {
            'answer': answer,
            'context': context,
            'question': question
        }
    
    except Exception as e:
        print(f'error is coming from answer question from context :: {e}')


def read_pdf_to_string(path):
    try:
        """
        Read the pdf document from the specified path and return its content as a string.
        
        Args:
            path: file path
        
        Return:
            str: the concatenated text content fo all pages in the pdf documents.
        """
        doc = fitz.open(path)

        content = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            content += page.get_text()
        return content

    except Exception as e:
        print(f'error is raised from read pdf to string :: {e}')


def split_text_into_chunks_with_indices(text: str, chunk_size: int, chunk_overlap: int):
    try:
        
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(Document(page_content=chunk,
                                   metadata = {'index': len(chunks)}))
            start += chunk_size - chunk_overlap

        return chunks


    except Exception as e:
        print(f'error is raised from split text into chunks with indices :: {e}')



def get_chunk_by_index(vectorstore, target_index:int) -> Document:
    try:
        """
        Retreive a chunk from the vectorstore based on its index in the metadata.

        Args:
            vectorstore: the vectorstore containing the chunks.
            target_index: the index of the chunk to be retrieved.
        
        Return:
            Document: the retreived chunk as a document object. or none.
        """

        all_docs = vectorstore.similarity_search("",
                                                 k = vectorstore.index.ntotal)
        for doc in all_docs:
            if doc.metadata.get('index') == target_index:
                return doc
        
    except Exception as e:
        print(f'error raised from get chunk by index function :: {e}')
