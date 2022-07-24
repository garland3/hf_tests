from transformers import pipeline
from getsampletxt import get_sample_txt
classifier = pipeline("zero-shot-classification")

txt = get_sample_txt(True)[1]
print(txt)
print("---"*10)

candidate_labels = ["politics", 'animals', 'education', 'religion', 'lifestyle', 'computers', 'health']
res = classifier(
    txt , 
    candidate_labels = candidate_labels,

    )
print(res)
scores = res['scores']
labels = res['labels']
for s,l in zip(scores,labels):
    print(f"{l}: {s}")
