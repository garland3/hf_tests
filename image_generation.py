# NOT task actuallly exists for this


# from transformers import pipeline
# from getsampletxt import get_sample_txt
# gen = pipeline("text-to-image")

# txt = get_sample_txt(True)

# print(txt)
# res = classifier(txt)
# print(res)

# !pip install diffusers
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline

model_id = "google/ddpm-cifar10-32"

# load model and scheduler
ddpm = DDPMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference

# run pipeline in inference (sample random noise and denoise)
image = ddpm()["sample"]


# save image
image[0].save("data/ddpm_generated_image.png")
