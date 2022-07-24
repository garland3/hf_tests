
# %%
from transformers import pipeline
# get audio example is here. 
# https://stackoverflow.com/a/35390981

from lib.capture_audio import capture_audio

audio  = capture_audio()
print(audio)
# %%
# import matplotlib.pyplot as plt


# # %%
# audio.shape
# # %%
# plt.plot(audio[:,0])
# # %%
# plt.plot(audio[:,1])

# # %%
# https://huggingface.co/tasks/automatic-speech-recognition
# classifier = pipeline("sentiment-analysis")

# with open("sample.flac", "rb") as f:
#   data = f.read()
# %%

# pipe = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
pipe = pipeline("automatic-speech-recognition")

# %%
single_channel = audio[:,0]
res = pipe(single_channel)
print(res)
# {'text': "GOING ALONG SLUSHY COUNTRY ROADS AND SPEAKING TO DAMP AUDIENCES IN DRAUGHTY SCHOOL ROOMS DAY AFTER DAY FOR A FORTNIGHT HE'LL HAVE TO PUT IN AN APPEARANCE AT SOME PLACE OF WORSHIP ON SUNDAY MORNING AND HE CAN COME TO US IMMEDIATELY AFTERWARDS"}


# %%
