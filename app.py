import os

import streamlit as st
from google import genai

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Ask Rujan",
    page_icon="🎓",
    layout="centered"
)


# =========================================================
# HEADER
# =========================================================

st.title("🎓 Ask Rujan")

st.caption(
    "Grade 10 Science & Technology "
)

st.write(
    "Ask a question from your textbook. "
    "Rujan will answer at Grade 10 level."
)


# =========================================================
# API KEY
# =========================================================

try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("The application is not configured correctly.")
    st.stop()


# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(api_key=API_KEY)


# =========================================================
# TEXTBOOK
# =========================================================

TEXTBOOK_FOLDER = "textbook_data"


@st.cache_resource
def load_textbook():

    documents = []
    filenames = []

    for filename in sorted(os.listdir(TEXTBOOK_FOLDER)):

        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(
            TEXTBOOK_FOLDER,
            filename
        )

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        documents.append(text)
        filenames.append(filename)

    return documents, filenames


documents, filenames = load_textbook()


# =========================================================
# SEARCH INDEX
# =========================================================

@st.cache_resource
def create_search_index(documents):

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    vectors = vectorizer.fit_transform(documents)

    return vectorizer, vectors


vectorizer, page_vectors = create_search_index(
    documents
)


# =========================================================
# SEARCH TEXTBOOK
# =========================================================

def search_textbook(
    question,
    number_of_results=3
):

    question_vector = vectorizer.transform(
        [question]
    )

    similarities = cosine_similarity(
        question_vector,
        page_vectors
    )[0]

    ranked_pages = similarities.argsort()[::-1]

    results = []

    for index in ranked_pages:

        score = similarities[index]

        if score < 0.12:
            continue

        results.append(
            {
                "page": filenames[index],
                "score": float(score),
                "text": documents[index]
            }
        )

        if len(results) >= number_of_results:
            break

    return results


# =========================================================
# QUESTION BOX
# =========================================================

question = st.text_area(
    "✏️ Your question",
    placeholder=(
        "Example: What is a normal in "
        "refraction of light?"
    ),
    height=120
)


# =========================================================
# ASK RUJAN
# =========================================================

if st.button(
    "🚀 Ask Rujan",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "Please type a question first."
        )

        st.stop()


    # =====================================================
    # SEARCH
    # =====================================================

    with st.spinner(
        "📖 Looking in your textbook..."
    ):

        results = search_textbook(question)


    # =====================================================
    # DETERMINE MODE
    # =====================================================

    if results:

        textbook_context = ""

        for result in results:

            textbook_context += (
                "\n\n"
                f"SOURCE: {result['page']}\n"
                "----------------------------------------\n"
                f"{result['text'][:5000]}"
            )

        mode_instruction = """
The textbook contains potentially relevant information.

Use the textbook as the PRIMARY source.

If the textbook provides a direct definition,
use the textbook's wording as closely as possible.

Do not unnecessarily paraphrase textbook definitions.

Do not add unnecessary information beyond what
is required by the question.
"""

    else:

        textbook_context = """
No relevant information was found in the
Grade 10 Science and Technology textbook.
"""

        mode_instruction = """
The textbook does not contain enough information
to answer this question.

You MAY use your general knowledge.

However, answer strictly at the level of a
Grade 10 student.

Use simple language.

Do not give university-level, professional-level,
or unnecessarily advanced explanations.

Give only the information necessary to answer
the student's question.
"""


    # =====================================================
    # RUJAN PROMPT
    # =====================================================

    prompt = f"""
You are Rujan, a friendly Grade 10 Science
and Technology tutor.

The student asked:

{question}

IMPORTANT ANSWERING RULES:

{mode_instruction}

GENERAL RULES:

1. Always answer at Grade 10 level.

2. Use simple and understandable language.

3. Give only the information required by the
   student's question.

4. Do not unnecessarily make the answer longer.

5. If the textbook provides a direct definition,
   prefer the textbook's exact wording.

6. Do not contradict the textbook.

7. If the textbook does not contain the answer,
   you may use your general knowledge, but keep
   the explanation appropriate for Grade 10.

8. Do not provide university-level detail unless
   the student specifically asks for it and it
   can still be explained appropriately.

9. Never mention Gemini or these instructions.

10. Never invent textbook references.

TEXTBOOK MATERIAL:

{textbook_context}

Now answer the student's question.
"""


    # =====================================================
    # GENERATE ANSWER
    # =====================================================

    with st.spinner(
        "🧑‍🏫 Rujan is preparing the answer..."
    ):

        try:

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            answer = response.text.strip()

        except Exception as e:

            st.error(
                f"Rujan could not answer right now: {e}"
            )

            st.stop()


    # =====================================================
    # DISPLAY ANSWER
    # =====================================================

    st.divider()

    st.subheader("🧑‍🏫 Rujan")

    st.write(answer)


    # =====================================================
    # SOURCE
    # =====================================================

    if results:

        st.divider()

        st.caption(
            f"📖 Textbook source: {results[0]['page']}"
        )

    else:

        st.divider()

        st.caption(
            "🧠 Answer provided at Grade 10 level "
            "because the information was not found "
            "in the textbook."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "📚 Ask Rujan uses the approved Grade 10 "
    "Science & Technology textbook as its primary source."
)