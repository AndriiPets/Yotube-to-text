from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PreProcessor, Crawler


def initialize_store() -> tuple[InMemoryDocumentStore, PreProcessor]:
    document_store = InMemoryDocumentStore()
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=200,
        split_respect_sentence_boundary=True,
    )

    return document_store, preprocessor
