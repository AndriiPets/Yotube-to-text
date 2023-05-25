import torch
from transformers import pipeline
import re

MODEL_NAME = "openai/whisper-tiny"

device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

all_special_ids = pipe.tokenizer.all_special_ids
transcribe_token_id = all_special_ids[-5]
translate_token_id = all_special_ids[-6]


def skip_tokens(text: str) -> str:
    pattern = r"(<.*?>)"
    return re.sub(pattern, "", text)


def transcribe(audio_path: str, task="transcribe"):
    print("transcribing")
    pipe.model.config.forced_decoder_ids = [
        [2, transcribe_token_id if task == "transcribe" else translate_token_id]
    ]
    text = pipe(audio_path)["text"]
    print("done")
    return skip_tokens(text)
