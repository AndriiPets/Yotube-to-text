from pytube import YouTube
import os

PATH = "./audio/audio.mp3"


def download_from_url(url: str) -> str:
    if os.path.exists(PATH):
        return PATH
    yt = YouTube(url)

    video = yt.streams.filter(only_audio=True).first()

    out_file = video.download(filename=PATH)

    return out_file
