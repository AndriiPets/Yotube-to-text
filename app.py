from utils.download_video import download_from_url
from utils.transcribe_audio import transcribe, api_transcribe, api_async_transcribe
from utils.summariser import summarize, api_summarize, llm_summarize
from qa_generator import generate_qa
import asyncio


URL = "https://www.youtube.com/watch?v=U3aXWizDbQ4"


async def main():
    await summarize_video_from_url(URL)
    # generate_questions(URL)


async def summarize_video_from_url(url):
    video = download_from_url(url)
    text = transcribe(video)
    tldr = llm_summarize(text)
    print(tldr)


def generate_questions(url):
    video = download_from_url(url)
    text = transcribe(video)
    qa = generate_qa(text)


if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
