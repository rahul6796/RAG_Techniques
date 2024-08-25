
from src.all_rag_techniques.simple_rag import encode_pdf
from src.all_rag_techniques.simple_rag import retrieve_context_per_question
from src.all_rag_techniques.simple_rag import path
from utils import show_context
from src.all_rag_techniques.context_enrichment_window_around_chunk import *



def main():
    # chunk_vector_store = encode_pdf(path=path,
    #                                 chunk_size=1000,
    #                                 chunk_overlap=200)
    
    # # create retriever
    # chunk_query_retriever = chunk_vector_store.as_retriever(
    #     search_kwargs = {'k':2}

    # )

    # test_query = "What is the main cause of climate change?"
    # context = retrieve_context_per_question(test_query, chunk_query_retriever)
    # show_context(context)
    
    query = "Explain the role of deforestation and fossil fuels in climate change."
    
    baseline_chunk = chunks_query_retriever.get_relevant_documents(query, k=1)

    enriched_chunk = retrieve_with_context_overlap(
        vectorstore,
        chunks_query_retriever,
        query,
        num_neighbors=1,
        chunk_size=400,
        chunk_overlap=200
    )

    print("Baseline Chunk:")
    print(baseline_chunk[0].page_content)
    print("\nEnriched Chunks:")
    print(enriched_chunk[0])


if __name__ == "__main__":
    main()

