from haystack.nodes import TransformersSummarizer
from haystack import Document

MODEL = "google/pegasus-xsum"


def summarize(inputs: str, *args) -> str:
    print("summarizing")
    docs = [Document(inputs)]
    summarizer = TransformersSummarizer(model_name_or_path=MODEL)
    summary = summarizer.predict(documents=docs)
    return summary[0].meta["summary"]
