from utils.document_store import initialize_store
from haystack.nodes import QuestionGenerator, FARMReader
from utils.file_to_store import text_to_store
from haystack.pipelines import QuestionAnswerGenerationPipeline
from tqdm.auto import tqdm
from haystack.utils import print_questions

MODEL = "deepset/minilm-uncased-squad2"


def generate_qa(text):
    doc_store, preprocessor = initialize_store()
    text_to_store(text, doc_store, preprocessor)
    reader = FARMReader(MODEL)
    question_generator = QuestionGenerator(split_length=200)
    qag_pipeline = QuestionAnswerGenerationPipeline(question_generator, reader)
    for idx, document in enumerate(tqdm(doc_store)):
        print(
            f"\n * Generating questions and answers for document {idx}: {document.content[:100]}...\n"
        )
        result = qag_pipeline.run(documents=[document])
        print_questions(result)
