# %%
from cProfile import label
from pathlib import Path
from typing import List
from transformers import pipeline
import cv2
from PIL import Image

# %%
from lib.get_images import get_images_folder, get_phone_images_as_imgs(), get_images_folder

# img = get_single_frame()
img = get_phone_images_as_imgs()[0]


# %%
# RUN THE MODEL
model = pipeline("image-segmentation")
# %%

res = model(img)
print(f"response is {res}")
# %%
# %%
images_folder = get_images_folder()
img.save(images_folder/ "original.jpg")
# %%
# res is a list of dictionaries, each with a 'score, label, and mask (as a PIL image)
for label_res in res:
    filename = images_folder / f"{label_res['label']}.jpg"
    mask_img = label_res['mask']
    mask_img.save(filename)

 

# %%
