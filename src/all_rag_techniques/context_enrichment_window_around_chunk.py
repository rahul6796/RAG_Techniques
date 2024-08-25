
import os
import sys

from langchain.docstore.document import Document
from langchain_community.embeddings import OllamaEmbeddings
from utils import read_pdf_to_string
from langchain_community.vectorstores import FAISS
from utils import split_text_into_chunks_with_indices
from utils import get_chunk_by_index
from typing import List

path = "./data/Understanding_Climate_Change.pdf"


content = read_pdf_to_string(path=path)

chunk_size = 400
chunk_overlap = 200

docs = split_text_into_chunks_with_indices(text=content,
                                           chunk_size=chunk_size,
                                           chunk_overlap=chunk_overlap)

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



ollama_emb = OllamaEmbeddings(
                model='nomic-embed-text', 
                show_progress=True  
            )

vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=ollama_emb
)



chunk = get_chunk_by_index(vectorstore,0)

chunks_query_retriever = vectorstore.as_retriever(search_kwargs={"k": 1})



def retrieve_with_context_overlap(vectorstore, retriever, query: str, num_neighbors: int = 1, chunk_size: int = 200, chunk_overlap: int = 20) -> List[str]:
    try:
        """
        Retrieve chunks based on a query, then fetch neighboring chunks and concatenate them, 
        accounting for overlap and correct indexing.

        Args:
        vectorstore (VectorStore): The vectorstore containing the chunks.
        retriever: The retriever object to get relevant documents.
        query (str): The query to search for relevant chunks.
        num_neighbors (int): The number of chunks to retrieve before and after each relevant chunk.
        chunk_size (int): The size of each chunk when originally split.
        chunk_overlap (int): The overlap between chunks when originally split.

        Returns:
        List[str]: List of concatenated chunk sequences, each centered on a relevant chunk.
        """
        relevant_chunks = retriever.get_relevant_documents(query)
        result_sequences = []

        for chunk in relevant_chunks:
            current_index = chunk.metadata.get('index')
            if current_index is None:
                continue

            # Determine the range of chunks to retrieve
            start_index = max(0, current_index - num_neighbors)
            end_index = current_index + num_neighbors + 1  # +1 because range is exclusive at the end

            # Retrieve all chunks in the range
            neighbor_chunks = []
            for i in range(start_index, end_index):
                neighbor_chunk = get_chunk_by_index(vectorstore, i)
                if neighbor_chunk:
                    neighbor_chunks.append(neighbor_chunk)

            # Sort chunks by their index to ensure correct order
            neighbor_chunks.sort(key=lambda x: x.metadata.get('index', 0))

            # Concatenate chunks, accounting for overlap
            concatenated_text = neighbor_chunks[0].page_content
            for i in range(1, len(neighbor_chunks)):
                current_chunk = neighbor_chunks[i].page_content
                overlap_start = max(0, len(concatenated_text) - chunk_overlap)
                concatenated_text = concatenated_text[:overlap_start] + current_chunk

            result_sequences.append(concatenated_text)

        return result_sequences
    
    except Exception as e:
        print(f'{e}')




