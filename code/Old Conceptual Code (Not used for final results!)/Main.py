from retrieval.DefaultPdfEmbedding import DefaultPdfEmbedder
from llm.LLM_Query import LLMQuery

def main(question, pdf_paths, retrieval_model_id, llm_model_id, k=1, chunk_size=100, metric='cosine'):
    embedder = DefaultPdfEmbedder(model_name=retrieval_model_id)
    llm = LLMQuery(model_id=llm_model_id)

    all_top_chunks = []
    
    for pdf_path in pdf_paths:
        top_chunks = embedder.retrieve_top_k_chunks(pdf_path, question, chunk_size=chunk_size, k=k, metric=metric)
        all_top_chunks.extend([chunk for chunk, _ in top_chunks])
    
    combined_chunks = "\n".join(all_top_chunks)
    
    answer = llm.query_llm(combined_chunks, question)
    print(answer[-1]["content"])

if __name__ == "__main__":
    question = 'What is the maximum number of attempts for a single exam?'
    pdf_paths = ['docs/SPO_Sample.pdf'] 
    retrieval_model_id = "mixedbread-ai/mxbai-embed-large-v1"
    llm_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    k = 10
    chunk_size = 100
    metric = 'hamming'

    main(question, pdf_paths, retrieval_model_id, llm_model_id, k=k, chunk_size=chunk_size, metric=metric)
