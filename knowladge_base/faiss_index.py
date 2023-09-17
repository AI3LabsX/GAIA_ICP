import logging
import os
import time
import typing as t
import uuid
from pathlib import Path

import openai
import tiktoken
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from utils import PROJECT_ROOT
from utils.env import env

logger = logging.getLogger(__name__)
openai.api_key = env.get_openai_api()


def tiktoken_len(text: str) -> int:
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


def process_recursive(documents: t.List[Document], embeddings: OpenAIEmbeddings) -> FAISS:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=tiktoken_len,
        separators=['\n\n', '\n', ' ', '']
    )
    chunked_documents = text_splitter.split_documents(documents)
    db = FAISS.from_documents(documents=chunked_documents, embedding=embeddings)
    return db


def get_data(sources: Path | str, embeddings: OpenAIEmbeddings, index_name) -> None:
    documents = []
    for file in os.listdir(sources):
        logger.info(f"Processing file {file}")
        if file.lower().endswith('.pdf'):
            try:
                pdf_path = Path(sources).joinpath(str(file))
                loader = PyMuPDFLoader(str(pdf_path))
                documents.extend(loader.load())

            except Exception as e:
                logger.error(f"Error processing file {file}: {e}")

    chunk_size = 15
    document_chunks = [documents[i:i + chunk_size] for i in range(0, len(documents), chunk_size)]
    print(len(document_chunks))

    main_db = process_recursive(document_chunks[0], embeddings)

    for chunk in document_chunks[1:]:
        logger.info("Processing chunk with documents:")
        for document in chunk:
            logger.info(
                f"Document '{document.metadata['source']}' contains {len(document.page_content.split())} tokens.")
        chunk_db = process_recursive(chunk, embeddings)
        main_db.merge_from(chunk_db)
        time.sleep(15)

    main_db.save_local(index_name)


if __name__ == '__main__':
    data = PROJECT_ROOT.joinpath('data')
    embeddings = OpenAIEmbeddings()
    ids = str(uuid.uuid4())
    get_data(data, embeddings, f'faiss_index_ICP')
