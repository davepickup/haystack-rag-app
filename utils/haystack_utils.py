"""Utility functions for buliding Haystack pipeline"""
import os

import streamlit as st
from datasets import load_dataset
from haystack.document_stores import BaseDocumentStore, InMemoryDocumentStore
from haystack.nodes import (  # noqa
    AnswerParser,
    BM25Retriever,
    PromptNode,
    PromptTemplate,
)
from haystack.pipelines import Pipeline

openai_api_key = os.getenv("OPENAI_KEY", None)

dataset = load_dataset("bilgeyucel/seven-wonders", split="train")


@st.cache_resource(show_spinner=False)
def start_document_store():
    """Instantiate Document store"""

    document_store = InMemoryDocumentStore(use_bm25=True)

    return document_store


@st.cache_resource(show_spinner=False)
def start_haystack_rag(_document_store: BaseDocumentStore):
    """Create Haystack RAG pipeline

    Parameters
    ----------
    _document_store : haystack.document_stores.memory.InMemoryDocumentStore
        Haystack DocumentStore object

    Returns
    ---------
    pipe : haystack.pipelines.base.Pipeline
        Haystack RAG pipeline object"""
    _document_store.write_documents(dataset)

    retriever = BM25Retriever(document_store=_document_store, top_k=2)

    rag_prompt = PromptTemplate(
        prompt="""Synthesize a comprehensive answer from the following text
        for the given question. Provide a clear and concise response that
        summarizes the key points and information presented in the text.
        Your answer should be in your own words and be no longer than 50 words.
        \n\n Related text: {join(documents)} \n\n Question: {query} \n\n Answer:""",  # noqa
        output_parser=AnswerParser(),
    )

    prompt_node = PromptNode(
        model_name_or_path="text-davinci-003",
        api_key=openai_api_key,
        default_prompt_template=rag_prompt,
    )
    pipe = Pipeline()

    pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
    pipe.add_node(
        component=prompt_node, name="PromptNode", inputs=["Retriever"]
    )  # noqa
    print(type(pipe))

    return pipe


@st.cache_data(show_spinner=True)
def query(_pipeline, question):
    """Run pipeline and return results

    Parameters
    ----------
    _pipeline : haystack.pipelines.base.Pipeline
        Haystack RAG pipeline object
    question : str
        User input question.

    Returns
    ---------
    results : dict
        RAG pipeline results.
    """

    params = {}
    results = _pipeline.run(question, params=params)
    print(type(results))
    return results
