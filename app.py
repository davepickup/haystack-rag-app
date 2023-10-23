import logging
import os
from json import JSONDecodeError

import streamlit as st

from utils.haystack_utils import query, start_document_store, start_haystack_rag  # noqa
from utils.streamlit_utils import reset_results, set_initial_state

try:
    document_store = start_document_store()
    pipeline = start_haystack_rag(document_store)

    set_initial_state()

    # Search bar
    question = st.text_input(
        "Ask a question",
        value=st.session_state.question,
        max_chars=100,
        on_change=reset_results,
    )

    run_pressed = st.button("Run")

    run_query = run_pressed or question != st.session_state.question

    # Get results for query
    if run_query and question:
        reset_results()
        st.session_state.question = question
        with st.spinner("🔎 &nbsp;&nbsp; Running your pipeline"):
            try:
                st.session_state.results = query(pipeline, question)
            except JSONDecodeError as je:
                logging.exception(je)
                st.error(
                    "👓 &nbsp;&nbsp; An error occurred reading the results. Is the document store working?"  # noqa
                )  # noqa
            except Exception as e:
                logging.exception(e)
                st.error("🐞 &nbsp;&nbsp; An error occurred during the request.")  # noqa

    if st.session_state.results:
        results = st.session_state.results
        st.write(results["answers"][0].answer)

except SystemExit as e:
    os._exit(e.code)
