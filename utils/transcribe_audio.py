import torch
from transformers import pipeline
from utils.timer_decorator import timeit
from decouple import config
from pydub import AudioSegment
from pydub.utils import make_chunks
import re
import requests
import json
import asyncio
import aiohttp

MODEL_NAME = "openai/whisper-tiny"
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v2"
API_KEY = config("HF_API_KEY")
headers = {"Authorization": f"Bearer {API_KEY}"}

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


@timeit
def transcribe(audio_path: str, task="transcribe"):
    prepare_audio(audio_path)
    pipe.model.config.forced_decoder_ids = [
        [2, transcribe_token_id if task == "transcribe" else translate_token_id]
    ]
    text = pipe(audio_path)["text"]
    return skip_tokens(text)


def prepare_audio(audio_path: str):
    sound_file = AudioSegment.from_file(audio_path)
    chunk_length = 30000
    chunks = make_chunks(sound_file, chunk_length=chunk_length)
    print(chunks)
    data = []
    for i, chunk in enumerate(chunks):
        chunk_name = f"audio\\chunks\\chunk{i}.mp3"
        chunk.export(chunk_name, format="mp3")


@timeit
async def api_async_transcribe(audio_path: str):
    with open(audio_path, "rb") as f:
        data = f.read()
        async with aiohttp.ClientSession() as session:
            async with session.post(API_URL, headers=headers, data=data) as url:
                response = await url.json()
    return response


def api_transcribe(audio_path: str):
    with open(audio_path, "rb") as f:
        data = f.read()
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))
