from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.quantization import quantize_embeddings
import numpy as np
import PyPDF2

class MrlPdfEmbedder:
    def __init__(self, model_name="mixedbread-ai/mxbai-embed-large-v1", dimensions=512):
        self.model = SentenceTransformer(model_name, truncate_dim=dimensions)
        self.dimensions = dimensions

    def read_pdf(self, file_path):
        pdf_text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                pdf_text += page.extract_text()
        return pdf_text

    def chunk_text(self, text, chunk_size=100):
        words = text.split()
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def encode(self, texts):
        return self.model.encode(texts)

    def quantize(self, embeddings, precision="ubinary"):
        return quantize_embeddings(embeddings, precision=precision)

    def hamming_distance_from_uint8(self, a, b):
        r = (1 << np.arange(8))[:, None]
        return np.count_nonzero((a & r) != (b & r))

    def retrieve_top_k_chunks(self, pdf_path, query, chunk_size=100, k=1, metric='cosine', apply_binarization=False):
        pdf_content = self.read_pdf(pdf_path)
        chunks = self.chunk_text(pdf_content, chunk_size=chunk_size)
        docs = [query] + chunks
        embeddings = self.encode(docs)

        if apply_binarization:
            embeddings = self.quantize(embeddings, precision="ubinary")

        query_embedding = embeddings[0]
        chunk_embeddings = embeddings[1:]

        if apply_binarization:
            distances = [self.hamming_distance_from_uint8(query_embedding, emb) for emb in chunk_embeddings]
            top_k_indices = np.argsort(distances)[:k]
        else:
            similarities = cos_sim(query_embedding, chunk_embeddings)[0]
            top_k_indices = np.argsort(-similarities)[:k]

        top_k_chunks = [(chunks[idx], distances[idx] if apply_binarization else similarities[idx].item()) for idx in top_k_indices]

        return top_k_chunks

    def get_model_for_mteb(self, apply_binarization=False):
        if apply_binarization:
            class BinarizedModel:
                def __init__(self, model, quantize_func):
                    self.model = model
                    self.quantize_func = quantize_func
                
                def encode(self, texts):
                    embeddings = self.model.encode(texts)
                    return self.quantize_func(embeddings, precision="ubinary")
            
            return BinarizedModel(self.model, self.quantize)
        else:
            return self.model
