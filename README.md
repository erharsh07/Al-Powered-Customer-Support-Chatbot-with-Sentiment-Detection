# AI-Powered Customer Support Chatbot with Sentiment Detection

A B.Tech CSE project demonstrating an AI customer-support chatbot using TF-IDF + Logistic Regression for intent classification and VADER for sentiment detection.

## Features
- Flask web application
- Intent classification
- Sentiment detection
- Confidence and sentiment scores
- Responsive chatbot UI
- Quick customer-support questions

## Supported intents
Greeting, Order Status, Refund, Payment, Account, Technical Support, Cancellation, Thanks.

## Run locally
```bash
python -m pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:5000` in your browser.

## Example questions
- Where is my order?
- I want a refund
- My payment failed
- I cannot login
- The app is not working
- I want to cancel my order
- Thank you

## Project structure
```text
AI-Customer-Support-Chatbot/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── templates/index.html
└── static/
    ├── style.css
    └── script.js
```

This is an academic/demo chatbot and does not connect to a real order, account, or payment database.
