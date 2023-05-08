from utils.download_video import download_from_url
from utils.transcribe_audio import transcribe

URL = "https://www.youtube.com/watch?v=U3aXWizDbQ4"


def main():
    video = download_from_url(URL)
    text = transcribe(video)
    print(text)


if __name__ == "__main__":
    main()
