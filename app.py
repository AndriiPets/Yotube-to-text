from utils.download_video import download_from_url
from utils.transcribe_audio import transcribe
from utils.summariser import summarize
from qa_generator import generate_qa

URL = "https://www.youtube.com/watch?v=U3aXWizDbQ4"


def main():
    summarize_video_from_url(URL)
    # generate_questions(URL)


def summarize_video_from_url(url):
    video = download_from_url(url)
    text = transcribe(video)
    tldr = summarize(text)
    print(tldr)


def generate_questions(url):
    video = download_from_url(url)
    text = transcribe(video)
    qa = generate_qa(text)


if __name__ == "__main__":
    main()
