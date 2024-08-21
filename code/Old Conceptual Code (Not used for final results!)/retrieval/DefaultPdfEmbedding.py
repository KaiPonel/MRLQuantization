from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.quantization import quantize_embeddings
import numpy as np
import PyPDF2

class DefaultPdfEmbedder:
    """
    A class to handle the embedding of PDF documents and retrieve relevant chunks of text based on a query.
    
    Attributes:
        model (SentenceTransformer): The sentence transformer model used for embedding.
        dimensions (int): The number of dimensions for the embeddings.
        quantization_method (str): The method used for quantizing embeddings.
    """

    def __init__(self, model_name="mixedbread-ai/mxbai-embed-large-v1", quantization_method="float32", dimensions=512):
        """
        Initializes the DefaultPdfEmbedder with a specified model, quantization method, and dimensions.

        Args:
            model_name (str): The name of the model to be used.
            quantization_method (str): The method for quantization ('float32', 'ubinary', 'uint8').
            dimensions (int): The number of dimensions for the embeddings.
        """
        self.model = SentenceTransformer(model_name, trust_remote_code=True, truncate_dim=dimensions)
        self.dimensions = dimensions
        self.quantization_method = quantization_method

    def read_pdf(self, file_path):
        """
        Reads a PDF file and extracts the text.

        Args:
            file_path (str): The path to the PDF file.

        Returns:
            str: The extracted text from the PDF.
        """
        pdf_text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                pdf_text += page.extract_text()
        return pdf_text

    def chunk_text(self, text, chunk_size=100):
        """
        Splits text into chunks of a specified size.

        Args:
            text (str): The text to be chunked.
            chunk_size (int): The number of words in each chunk.

        Returns:
            list of str: The text divided into chunks.
        """
        words = text.split()
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def encode(self, sentences, quantize=True, **kwargs):
        """
        Encodes sentences into embeddings, optionally quantizing them.

        Args:
            sentences (list of str): The sentences to be encoded.
            quantize (bool): Whether to quantize the embeddings.

        Returns:
            numpy.ndarray: The embeddings of the sentences.
        """
        if quantize:
            return self.quantize(self.model.encode(sentences))
        return self.model.encode(sentences)
    
    def quantize(self, embeddings):
        """
        Quantizes the embeddings using the specified precision.

        Args:
            embeddings (numpy.ndarray): The embeddings to be quantized.

        Returns:
            numpy.ndarray: The quantized embeddings.
        """
        return quantize_embeddings(embeddings, precision=self.quantization_method)

    def hamming_distance_from_uint8(self, a, b):
        """
        Computes the Hamming distance between two uint8 embeddings.

        Args:
            a (numpy.ndarray): The first embedding.
            b (numpy.ndarray): The second embedding.

        Returns:
            int: The Hamming distance between the two embeddings.
        """
        r = (1 << np.arange(8))[:, None]
        return np.count_nonzero((a & r) != (b & r))

    def retrieve_top_k_chunks(self, pdf_path, query, chunk_size=100, k=1, metric='cosine'):
        """
        Retrieves the top-k chunks of text from a PDF that are most relevant to a query.

        Args:
            pdf_path (str): The path to the PDF file.
            query (str): The query to search for in the PDF.
            chunk_size (int): The number of words in each chunk.
            k (int): The number of top chunks to retrieve.
            metric (str): The similarity metric to use ('cosine' or 'hamming').

        Returns:
            list of tuple: The top-k chunks and their similarity scores.
        """
        pdf_content = self.read_pdf(pdf_path)
        chunks = self.chunk_text(pdf_content, chunk_size=chunk_size)
        
        docs = [query] + chunks
        embeddings = self.encode(docs)

        embeddings = self.quantize(embeddings)

        query_embedding = embeddings[0]
        chunk_embeddings = embeddings[1:]

        if self.quantization_method == "ubinary":
            distances = [self.hamming_distance_from_uint8(query_embedding, emb) for emb in chunk_embeddings]
            top_k_indices = np.argsort(distances)[:k]
        else:
            similarities = cos_sim(query_embedding, chunk_embeddings)[0]
            top_k_indices = np.argsort(-similarities)[:k]

        top_k_chunks = []
        for idx in top_k_indices:
            if self.quantization_method == "ubinary":
                top_k_chunks.append((chunks[idx], distances[idx]))
            else:
                top_k_chunks.append((chunks[idx], similarities[idx].item()))

        return top_k_chunks