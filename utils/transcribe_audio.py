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
import os
import base64

MODEL_NAME = "openai/whisper-tiny"
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v2"
API_URL_BACKUP = (
    "https://api-inference.huggingface.co/models/facebook/wav2vec2-base-960h"
)
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
    directory = "audio\\chunks"
    for i, chunk in enumerate(chunks):
        chunk_name = f"audio\\chunks\\chunk{i}.mp3"
        chunk.export(chunk_name, format="mp3")
    return directory


@timeit
async def api_async_transcribe(audio_path: str) -> list[str]:
    for _ in range(4):
        print(f"Transcribing from API attempt: {_}")
        try:
            api_response = await query_api(audio_path, API_URL_BACKUP)
            transcription = api_response["text"].lower()

            return transcription
        except:
            if "error" in api_response and "estimated_time" in api_response:
                wait_time = api_response["estimated_time"]
                if int(wait_time) > 20:
                    continue
                else:
                    print("Waiting for model to load....", wait_time)
                    # waiting for the model to load + 5sec
                    await asyncio.sleep(wait_time + 5.0)
            elif "error" in api_response:
                raise RuntimeError("Error Fetching API", api_response["error"])
            else:
                break


async def query_api(audio_path: str, model_url: str):
    with open(audio_path, "rb") as f:
        payload = json.dumps(
            {
                "inputs": base64.b64encode(f.read()).decode("utf-8"),
                "parameters": {
                    "return_timestamps": "char",
                    "chunk_length_s": 10,
                    "stride_length_s": [4, 2],
                },
                "options": {"use_gpu": False},
            }
        ).encode("utf-8")
    async with aiohttp.ClientSession() as session:
        async with session.post(model_url, headers=headers, data=payload) as url:
            response = await url.json()
    return response


def api_transcribe(audio_path: str):
    with open(audio_path, "rb") as f:
        data = f.read()
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))
