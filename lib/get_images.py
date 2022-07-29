
# %%
from pathlib import Path
from typing import List
import cv2
from PIL import Image

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

def get_images_folder() -> Path:
    # MAKE somewhere to save the results
    data_folder = Path("data")
    data_folder.mkdir(exist_ok=True)
    images_folder = data_folder / "images"
    images_folder.mkdir(exist_ok=True)
    return images_folder

def get_phone_images_pathes() -> List:
    phone_folder = get_images_folder() / "phone"
    if phone_folder.exists() == False:
        return []
    return list(phone_folder.rglob("*.jpg"))

def get_phone_images_as_imgs()    :
    return [Image.open(f) for f in get_phone_images_pathes()]

