
from transformers import pipeline
from lib.getsampletxt import get_sample_txt
classifier = pipeline("sentiment-analysis")

txt = get_sample_txt(True)

print(txt)
res = classifier(txt)
print(res)

