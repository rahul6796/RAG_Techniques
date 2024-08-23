
from src.all_rag_techniques.simple_rag import encode_pdf
from src.all_rag_techniques.simple_rag import retrieve_context_per_question
from src.all_rag_techniques.simple_rag import path
from utils import show_context
from src.all_rag_techniques.context_enrichment_window_around_chunk import content


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
    print(content)
    


if __name__ == "__main__":
    main()

