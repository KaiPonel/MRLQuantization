import transformers
import torch

class LLMQuery:
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.pad_token_id = self.pipeline.tokenizer.pad_token_id if self.pipeline.tokenizer.pad_token_id is not None else self.pipeline.tokenizer.eos_token_id

    def generate_rag_prompt_content(self, document_info, question):
        return (
            "DOCUMENT:\n{}\nQUESTION:\n{}\nINSTRUCTIONS: Answer the user's QUESTION "
            "using the DOCUMENT text above. Keep your answer grounded in the facts of the DOCUMENT. "
            "If the DOCUMENT doesn’t contain the facts to answer the QUESTION, return "
            "'The documents do not contain the information to answer the question.'".format(document_info, question)
        )

    def query_llm(self, document_info, question):
        query = self.generate_rag_prompt_content(document_info, question)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant who only uses information provided in documents. You are not allowed to use any other information which is not provided in the DOCUMENT section of each query. Using the Information in the Document section you have answer the question in the QUESTION section. If the information is not available in the document to answer the question, you should return 'The documents do not contain the information to answer the question.'"},
            {"role": "user", "content": query},
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=256,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.pipeline.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        
        generated_text = outputs[0]["generated_text"]
        return generated_text
