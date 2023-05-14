from haystack.nodes import TransformersSummarizer
from utils.file_to_store import text_to_store
from utils.document_store import initialize_store

MODEL = "facebook/bart-large-cnn"

doc_store, preprocessor = initialize_store()


def summarize(inputs: str, *args) -> str:
    print("summarizing")
    text_to_store(inputs, doc_store, preprocessor)
    summarizer = TransformersSummarizer(
        model_name_or_path=MODEL, max_length=200, progress_bar=True
    )
    summary = summarizer.predict(documents=doc_store.get_all_documents())
    return "\n".join([s.meta["summary"] for s in summary])
