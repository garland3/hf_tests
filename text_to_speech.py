# https://huggingface.co/facebook/fastspeech2-en-ljspeech

#  https://github.com/facebookresearch/fairseq
# git clone https://github.com/pytorch/fairseq
# cd fairseq
# pip install -e ./
# pip install g2p-en

# %%

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd

# %%
models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/fastspeech2-en-ljspeech",
    arg_overrides={"vocoder": "hifigan", "fp16": False}
)

# %%
model = models[0]

# %%
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator(models, cfg)

text = """Hi Aubri.
 I hope you are having fun playing on mindcraft. 
 Can you please make me a smoothy. Yogurt smoothies make my circuits run faster. """

text = """Hi Chicken. Can you add some more flowers? """
# %%
sample = TTSHubInterface.get_model_input(task, text)
wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)

# %%
# NOT working for me in VScode. 
ipd.Audio(wav, rate=rate)

# %%
from scipy.io.wavfile import write
# %%
wav_np = wav.numpy()
# %%

write('data/test.wav', rate, wav_np)
# %%
