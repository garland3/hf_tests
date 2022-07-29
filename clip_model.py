# %%

from PIL import Image
import requests
from lib.get_images import get_phone_images_as_imgs, get_phone_images_pathes

from transformers import CLIPProcessor, CLIPModel



model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# %%

image = get_phone_images_as_imgs()[0]
image.resize((500,500))
# %%



# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
text=["a photo of a cat", "a photo of a dog", "a girl", "a girl at the lake", "a boy with a fish", "a girl with a fish"]
inputs = processor(text = text, images=image, return_tensors="pt", padding=True)
# %%
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

# %%
import matplotlib.pyplot as plt
# %%
probs_np = probs.squeeze().detach().numpy()
# %%

plt.plot(probs_np)
# %%
for t,p in zip(text, probs_np):
    print(f"{p}: {t}")
# %%
