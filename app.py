from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

training_data = [
    ("hello", "greeting"), ("hi", "greeting"), ("hey", "greeting"),
    ("hello there", "greeting"), ("good morning", "greeting"),
    ("good evening", "greeting"), ("I need help", "greeting"),
    ("can you help me", "greeting"),
    ("where is my order", "order_status"), ("where is my package", "order_status"),
    ("track my order", "order_status"), ("what is the status of my order", "order_status"),
    ("my order has not arrived", "order_status"), ("can you track my delivery", "order_status"),
    ("when will my order arrive", "order_status"), ("order delivery status", "order_status"),
    ("I want a refund", "refund"), ("how can I get a refund", "refund"),
    ("my refund is not received", "refund"), ("when will I get my refund", "refund"),
    ("I need my money back", "refund"), ("refund status", "refund"),
    ("my payment failed", "payment"), ("why did my payment fail", "payment"),
    ("payment is not working", "payment"), ("I was charged twice", "payment"),
    ("I have a problem with payment", "payment"), ("payment failed during checkout", "payment"),
    ("I cannot login", "account"), ("I forgot my password", "account"),
    ("how do I reset my password", "account"), ("my account is locked", "account"),
    ("I have a problem with my account", "account"), ("unable to sign in", "account"),
    ("the app is not working", "technical_support"), ("website is not working", "technical_support"),
    ("I have a technical problem", "technical_support"), ("the system is showing an error", "technical_support"),
    ("something is not working", "technical_support"), ("the application has an error", "technical_support"),
    ("I want to cancel my order", "cancellation"), ("cancel my order", "cancellation"),
    ("how can I cancel my order", "cancellation"), ("I want to cancel my purchase", "cancellation"),
    ("please cancel my order", "cancellation"),
    ("thank you", "thanks"), ("thanks", "thanks"),
    ("thank you for helping me", "thanks"), ("that's helpful", "thanks"),
    ("thanks for your support", "thanks")
]

texts = [x[0] for x in training_data]
labels = [x[1] for x in training_data]
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
    ("classifier", LogisticRegression(max_iter=1000))
])
model.fit(texts, labels)
sentiment_analyzer = SentimentIntensityAnalyzer()

RESPONSES = {
    "greeting": "Hello! Welcome to customer support. How can I help you today?",
    "order_status": "Sure! I can help with order tracking. Please provide your order ID so the delivery status can be checked.",
    "refund": "I can help with your refund. Please provide your order ID and the reason for the refund.",
    "payment": "I'm sorry you're having trouble with payment. Please check your payment details and try again. If the issue continues, contact support with your transaction details.",
    "account": "I can help with your account. Please try the password-reset option on the login page. If your account is locked, contact support for assistance.",
    "technical_support": "I can help troubleshoot the technical issue. Please tell me what error you are seeing and which app or page is affected.",
    "cancellation": "I can help with cancellation. Please provide your order ID so the cancellation request can be checked.",
    "thanks": "You're welcome! I'm happy to help. Is there anything else you need?"
}

def detect_sentiment(message):
    score = sentiment_analyzer.polarity_scores(message)["compound"]
    if score >= 0.05:
        label = "Positive"
    elif score <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return label, score

def process_message(message):
    message = message.strip()
    if not message:
        return {"response": "Please type a question or message.", "intent": "unknown", "sentiment": "Neutral", "sentiment_score": 0.0, "confidence": 0.0}
    probabilities = model.predict_proba([message])[0]
    best_index = probabilities.argmax()
    intent = model.classes_[best_index]
    confidence = float(probabilities[best_index])
    sentiment, score = detect_sentiment(message)
    if confidence < 0.28:
        response = "I'm not completely sure I understood that. Try asking about an order, refund, payment, account, technical issue, or cancellation."
        intent = "unknown"
    else:
        response = RESPONSES[intent]
    return {"response": response, "intent": intent, "sentiment": sentiment, "sentiment_score": round(score, 3), "confidence": round(confidence * 100, 1)}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    return jsonify(process_message(data.get("message", "")))

if __name__ == "__main__":
    print("=" * 60)
    print("AI CUSTOMER SUPPORT CHATBOT")
    print("=" * 60)
    print("Server running at:")
    print("http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True)
