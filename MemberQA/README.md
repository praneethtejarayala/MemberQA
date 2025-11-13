🧠 Member Q&A API

A simple question-answering API that responds to natural-language questions about members based on their public messages.

🚀 Example
GET /ask?question=What%20are%20Amira's%20favorite%20restaurants?


Response:

{ "answer": "Amira loves Nobu and Din Tai Fung." }

⚙️ How It Works

Fetches all messages from the public /messages endpoint.

Identifies which member the question is about.

Uses TF-IDF vectorization + cosine similarity to find the most semantically similar message.

Returns that message as the answer.

🧩 Alternative Approaches (Bonus 1)
Approach	Description	Pros	Cons
TF-IDF + Cosine (current)	Simple and fast local similarity	No API cost, interpretable	Limited semantic depth
Sentence Embeddings (e.g., MiniLM, InstructorXL)	Use pretrained models for semantic similarity	Higher accuracy	Requires model + more memory
LLM-based (OpenAI, Gemini)	Use GPT-style models for reasoning	Handles complex logic	External dependency, API cost

📊 Data Insights (Bonus 2)

Some observed anomalies in the dataset:

Certain users have duplicate messages with near-identical text.

A few entries contain non-English or emoji-heavy content which reduces TF-IDF effectiveness.

Some user_name values are missing or inconsistent in casing (e.g., "Layla" vs "layla").

There are messages that appear to belong to group contexts, not individual members.

🧪 Run Locally
pip install -r requirements.txt
uvicorn app:app --reload


Then open http://localhost:8000/docs

🌍 Deployment

Deploy easily on Render, Railway, or Cloud Run using:

uvicorn app:app --host 0.0.0.0 --port $PORT