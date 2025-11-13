from fastapi import FastAPI, Query
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(
    title="Member Q&A API",
    description="Answers natural-language questions about members using TF-IDF similarity.",
    version="1.0.0",
)

MESSAGES_API = "https://november7-730026606190.europe-west1.run.app/messages/"


def clean_text(text: str) -> str:
    """Normalize text for better matching."""
    text = text.lower().strip()
    text = re.sub(r"http\S+|@\w+|#[\w-]+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text)


def find_member_name(question: str, member_names: list[str]) -> str | None:
    """Identify a member name in the question (basic match)."""
    q = question.lower()
    for name in member_names:
        if not name:
            continue
        n = name.lower()
        if n in q or n.split()[0] in q:
            return name
    return None


@app.get("/")
def root():
    return {"status": "ok", "message": "Member Q&A API is running."}


@app.get("/ask")
def ask(question: str = Query(..., description="Enter your question")):
    """Return the most relevant message for a given natural-language question."""
    try:
        res = requests.get(MESSAGES_API, timeout=10)
        res.raise_for_status()
        data = res.json()
        messages = data.get("items", [])
    except Exception as e:
        return {"answer": f"Could not fetch messages: {e}"}

    if not messages:
        return {"answer": "No dataset messages found."}

    question_clean = clean_text(question)

    # Identify member
    member_names = [m.get("user_name") for m in messages if m.get("user_name")]
    member_name = find_member_name(question_clean, member_names)

    if not member_name:
        return {"answer": "Could not identify the member in your question."}

    # Filter that member's messages
    member_msgs = [m for m in messages if m.get("user_name") == member_name]
    if not member_msgs:
        return {"answer": f"No messages found for {member_name}."}

    # Compute TF-IDF similarity
    texts = [question_clean] + [clean_text(m["message"]) for m in member_msgs]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    best_idx = sims.argmax()
    best_msg = member_msgs[best_idx]
    best_score = float(sims[best_idx])

    if best_score < 0.1:
        return {"answer": "No relevant info found."}

    return {"answer": best_msg["message"]}
