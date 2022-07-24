from transformers import pipeline
from getsampletxt import get_sample_txt

gen = pipeline("text-generation", model = 'distilgpt2')

txt = get_sample_txt(True)[1]
print(txt)
print("--"*20)
# res = classifier(txt)
res = gen(
    txt, 
    max_length = 2000,
    num_return_sequences = 1,
)
print(res)