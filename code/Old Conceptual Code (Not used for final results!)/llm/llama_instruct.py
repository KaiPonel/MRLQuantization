import transformers
import torch
import os 

# 8k Context Window
# conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def generate_rag_prompt_content(document_info, question):
    return "DOCUMENT:\n{}\n QUESTION:\n{}\n INSTRUCTIONS: Answer the users QUESTION using the DOCUMENT text above. Keep your answer ground in the facts of the DOCUMENT. If the DOCUMENT doesn’t contain the facts to answer the QUESTION return 'The documents do not contain the information to answer the question.'".format(document_info, question)

query = generate_rag_prompt_content("The sky is blue.", "What color is the mars?")


messages = [
    {"role": "system", "content": "You are a helpful assistant who only uses information provided in documents. You are not allowed to use any other information which is not provided in the DOCUMENT section of each query. Using the Information in the Document section you have answer the question in the QUESTION section. If the information is not available in the document to answer the question, you should return 'The documents do not contain the information to answer the question.'"},
    {"role": "user", "content": query},
]

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][-1])