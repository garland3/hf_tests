from transformers import pipeline
from getsampletxt import get_sample_txt

qa_model = pipeline("question-answering")
question = "Who is in trouble?"
context = get_sample_txt()[1]
res = qa_model(question = question, context = context)
## {'answer': 'İstanbul', 'end': 39, 'score': 0.953, 'start': 31}
print(res)
