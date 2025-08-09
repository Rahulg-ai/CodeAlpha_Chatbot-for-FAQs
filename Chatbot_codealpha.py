import tkinter as tk
from tkinter import scrolledtext
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing functions
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(filtered_tokens)

# Sample FAQs
faq_data = {
    "What is AI?": "Artificial Intelligence is the simulation of human intelligence in machines.",
    "What is machine learning?": "Machine learning is a subset of AI that allows systems to learn from data.",
    "What is supervised learning?": "Supervised learning is a type of ML where the model is trained on labeled data.",
    "What is unsupervised learning?": "Unsupervised learning uses unlabeled data to find hidden patterns.",
    "What is deep learning?": "Deep learning is a type of ML using neural networks with multiple layers.",
    "What is NLP?": "Natural Language Processing allows computers to understand and interpret human language.",
    "What is a chatbot?": "A chatbot is a software that can simulate a conversation with a human user.",
    "What is data preprocessing?": "Data preprocessing is the process of cleaning and preparing raw data for analysis.",
    "What is overfitting?": "Overfitting is when a model performs well on training data but poorly on new data.",
    "What is a neural network?": "A neural network is a set of algorithms designed to recognize patterns like the human brain."
}

questions = list(faq_data.keys())
answers = list(faq_data.values())

# Preprocess questions
preprocessed_questions = [preprocess(q) for q in questions]

# Vectorize FAQs
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(preprocessed_questions)

# Chatbot response function
def get_bot_response(user_input):
    user_input_processed = preprocess(user_input)
    user_vector = vectorizer.transform([user_input_processed])
    similarities = cosine_similarity(user_vector, question_vectors)
    max_sim_index = np.argmax(similarities)

    if similarities[0][max_sim_index] > 0.3:  # threshold to ensure relevant match
        return answers[max_sim_index]
    else:
        return "I'm sorry, I couldn't find a matching answer in our FAQ database. Please rephrase or contact support."

# GUI setup
window = tk.Tk()
window.title("AI FAQ Chatbot")
window.geometry("550x600")
window.configure(bg="#f5f5f5")

# Chat display area
chat_display = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=60, height=25, font=("Arial", 11))
chat_display.pack(padx=10, pady=10)
chat_display.config(state=tk.DISABLED)

# Display messages
def display_message(sender, message):
    chat_display.config(state=tk.NORMAL)
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    chat_display.insert(tk.END, f"{timestamp} {sender}: {message}\n")
    chat_display.config(state=tk.DISABLED)
    chat_display.see(tk.END)

# Handle send button
def send_message():
    user_msg = user_input.get()
    if user_msg.strip() == "":
        return
    display_message("You", user_msg)
    bot_response = get_bot_response(user_msg)
    display_message("Bot", bot_response)
    user_input.delete(0, tk.END)

# Handle quick question button click
def send_quick_question(q):
    user_input.delete(0, tk.END)
    user_input.insert(0, q)
    send_message()

# Clear chat
def clear_chat():
    chat_display.config(state=tk.NORMAL)
    chat_display.delete(1.0, tk.END)
    chat_display.config(state=tk.DISABLED)

# Quick question buttons
quick_frame = tk.Frame(window, bg="#f5f5f5")
quick_frame.pack(pady=5)

quick_questions = [
    "What is AI?",
    "What is machine learning?",
    "What is supervised learning?",
    "What is NLP?",
    "What is a chatbot?",
    "What is deep learning?"
]

for q in quick_questions:
    tk.Button(
        quick_frame, text=q, command=lambda q=q: send_quick_question(q),
        bg="#e0e0e0", font=("Arial", 9), relief=tk.RAISED, wraplength=120
    ).pack(side=tk.LEFT, padx=4, pady=5)

# Input frame
input_frame = tk.Frame(window, bg="#f5f5f5")
input_frame.pack(pady=10)

user_input = tk.Entry(input_frame, width=40, font=("Arial", 12))
user_input.pack(side=tk.LEFT, padx=5)

send_button = tk.Button(input_frame, text="Send", command=send_message, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
send_button.pack(side=tk.LEFT)

clear_button = tk.Button(input_frame, text="Clear", command=clear_chat, bg="#FF7F7F", fg="white", font=("Arial", 10, "bold"))
clear_button.pack(side=tk.LEFT, padx=5)

# Run the GUI loop
window.mainloop()
