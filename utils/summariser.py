from haystack.nodes import TransformersSummarizer
from haystack.nodes import PromptNode
from haystack.errors import HuggingFaceInferenceError
from utils.file_to_store import text_to_store
from utils.document_store import initialize_store
from decouple import config
import requests
import json
import aiohttp
import asyncio
from utils.timer_decorator import timeit
import time


MODEL = "facebook/bart-large-cnn"
LLM_MODEL = "OpenAssistant/oasst-sft-1-pythia-12b"
MODEL_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
API_KEY = config("HF_API_KEY")

headers = {"Authorization": f"Bearer {API_KEY}"}

doc_store, preprocessor = initialize_store()


@timeit
def summarize(inputs: str, *args) -> str:
    text_to_store(inputs, doc_store, preprocessor)
    summarizer = TransformersSummarizer(
        model_name_or_path=MODEL, max_length=200, progress_bar=True
    )
    summary = summarizer.predict(documents=doc_store.get_all_documents())
    return "\n".join([s.meta["summary"] for s in summary])


@timeit
def api_summarize(inputs: str) -> str:
    data = json.dumps(inputs)
    response = requests.request("POST", MODEL_API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


@timeit
def llm_summarize(inputs: str) -> str:
    text_to_store(inputs, doc_store, preprocessor)
    summary = "Failed request"
    for _ in range(3):
        try:
            prompt_node = PromptNode(
                model_name_or_path=LLM_MODEL, api_key=API_KEY, max_length=246
            )
            summary = prompt_node.prompt(
                prompt_template="summarization", documents=doc_store.get_all_documents()
            )
            break
        except HuggingFaceInferenceError as err:
            print(f"Error number: {err.status_code}")
            if err.status_code == "503":
                print("Wainting 10 seconds for model to load...")
        print(f"cycle number: {_}")
        time.sleep(10)
    return summary
