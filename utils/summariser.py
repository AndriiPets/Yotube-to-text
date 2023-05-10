from haystack.nodes import TransformersSummarizer
from haystack import Document
from utils.document_store import initialize_store

MODEL = "facebook/bart-large-cnn"

doc_store, preprocessor = initialize_store()


def text_to_store(text):
    doc_store.delete_documents()
    doc = [Document(text)]
    preprocessesd_docs = preprocessor.process(doc)
    doc_store.write_documents(preprocessesd_docs)


def summarize(inputs: str, *args) -> str:
    print("summarizing")
    text_to_store(inputs)
    summarizer = TransformersSummarizer(
        model_name_or_path=MODEL, max_length=200, progress_bar=True
    )
    summary = summarizer.predict(documents=doc_store.get_all_documents())
    return "\n".join([s.meta["summary"] for s in summary])
