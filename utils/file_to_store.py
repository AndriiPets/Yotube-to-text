from haystack import Document


def text_to_store(text, doc_store, preprocessor):
    doc_store.delete_documents()
    doc = [Document(text)]
    preprocessesd_docs = preprocessor.process(doc)
    doc_store.write_documents(preprocessesd_docs)
