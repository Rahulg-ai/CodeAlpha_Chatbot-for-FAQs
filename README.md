# 💬 CodeAlpha Chatbot for FAQs

An intelligent **Python chatbot** that answers Frequently Asked Questions (FAQs) using **Natural Language Processing (NLP)** with a friendly **Tkinter GUI**.
This project was developed as part of the **CodeAlpha Internship – Task 2**.

---

## 📌 Features

* 🖥 **User-Friendly GUI** built with Tkinter
* 🤖 **FAQ Matching** using **TF-IDF Vectorization** & **Cosine Similarity**
* 🗣 **NLP Preprocessing** with NLTK (tokenization, stopword removal, lemmatization)
* ⚡ Instant responses to predefined questions
* ❌ **Clear Chat** button to reset the conversation
* 🔍 **Quick Questions** section for fast queries

---

## 🛠️ Tech Stack

* **Python 3.x**
* **Tkinter** (GUI)
* **NLTK** – Natural Language Toolkit
* **scikit-learn** – Machine Learning utilities
* **NumPy** – Numerical computing

---

## 📂 Project Structure

```
📦 CodeAlpha_Chatbot
 ┣ 📜 Chatbot_codealpha.py   # Main chatbot script
 ┣ 📜 faq_dataset.json       # FAQ database
 ┣ 📜 README.md              # Project documentation
```

---

## 🚀 Installation & Usage

1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/CodeAlpha_Chatbot-for-FAQs.git
```

2️⃣ Install dependencies

```bash
pip install nltk scikit-learn numpy
```

3️⃣ Run the chatbot

```bash
python Chatbot_codealpha.py
```

4️⃣ Enjoy chatting! 💬

---

## 📸 Screenshot

(Add your screenshot here)
Example:
![Chatbot GUI](screenshot.png)

---

## 🧠 How It Works

1. User enters a question in the GUI.
2. The chatbot vectorizes the input and compares it with stored FAQs using **TF-IDF + Cosine Similarity**.
3. The closest matching answer is displayed in the chat window.

---

## 📌 Future Improvements

* 🌍 Multi-language support
* 🎤 Voice input & output
* ☁ Online deployment (Flask / Django backend)

---

## 👩‍💻 Author

Rahul Gupta
📧 *\[your email]*
🔗 [GitHub Profile](https://github.com/Rahulg-ai)
