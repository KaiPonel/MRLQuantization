from mteb import MTEB
from sentence_transformers import SentenceTransformer
from retrieval.DefaultPdfEmbedding import DefaultPdfEmbedder
import numpy as np

model_name= "dunzhang/stella_en_400M_v5"

mrl_models = [
    {"name": "dunzhang/stella_en_400M_v5", "dimensions": 8192},
    {"name": "mixedbread-ai/mxbai-embed-large-v1", "dimensions": 1024}
]

def test_all_models_base():
    for mrl_model in mrl_models:
        model = DefaultPdfEmbedder(model_name=mrl_model["name"], dimensions=mrl_model["dimensions"])
        evaluation = MTEB(tasks=["Banking77Classification"])
        results = evaluation.run(model, output_folder=f"results/{mrl_model['name']}_{mrl_model['dimensions']}")

def test_all_models_varying_dim(number_of_reductions=1):
    for mrl_model in mrl_models:
        for i in range(number_of_reductions):
            # If the dimensions are not divisible by 2 -> end
            if mrl_model["dimensions"] / (2 **(i-1)) % 2 != 0:
                break

            model = DefaultPdfEmbedder(model_name=mrl_model["name"], dimensions=mrl_model["dimensions"] / (2 **(i-1)))
            evaluation = MTEB(tasks=["Banking77Classification"])
            results = evaluation.run(model, output_folder=f"results/{mrl_model['name']}_{mrl_model['dimensions'] / (2 **(i-1))}")


#test_all_models_base()

model = DefaultPdfEmbedder(model_name="mixedbread-ai/mxbai-embed-large-v1", dimensions=1024, quantization_method="int8")

out = model.encode("Hello World").astype(np.float32)

print(len(out))
print(type(out[0]))

# ToDo: Cast to FP32. 
