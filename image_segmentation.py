# %%
from cProfile import label
from pathlib import Path
from transformers import pipeline
import cv2
from PIL import Image

model = pipeline("image-segmentation")

# %%
def get_single_frame(as_PIL = True):
    camera = cv2.VideoCapture(0)
    # while True:
    ret, frame = camera.read()
        # cv2.imshow('frame',frame)
        # if cv2.waitKey(1) == ord('q'):
            # break
    camera.release()
    cv2.destroyAllWindows()
    if as_PIL:
        return Image.fromarray(frame)
    return frame

# %%

img = get_single_frame()

# %%
# RUN THE MODEL
res = model(img)

# %%
# MAKE somewhere to save the results
data_folder = Path("data")
images_folder = data_folder / "images"
images_folder.mkdir(exist_ok=True)
# %%
img.save(images_folder/ "original.jpg")
# %%
# res is a list of dictionaries, each with a 'score, label, and mask (as a PIL image)
for label_res in res:
    filename = images_folder / f"{label_res['label']}.jpg"
    mask_img = label_res['mask']
    mask_img.save(filename)



# %%
