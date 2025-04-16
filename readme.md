# 📚 Intent-Aware ChatBot with LangGraph & LangChain

This project is a modular, intelligent chatbot built using **LangChain**, **LangGraph**, and **GPT-4o** that routes user queries based on intent—whether it’s a general conversation, checking the weather, getting the current time, generating code, fetching news, solving math, or translating text.

---

## 🚀 Features

- 🧠 **Intent Classification** — Smart routing to the correct skill based on user input.
- 💬 **General Chat** — Handles open-domain conversations.
- 🌦️ **Weather Assistant** — Gets current weather info for any city using OpenWeather API.
- ⏰ **Time Lookup** — Tells current time anywhere in the world using geolocation and timezone detection.
- 💻 **Code Assistant** — Generates well-commented code snippets.
- 📰 **News Fetcher** — Displays top headlines using NewsAPI.
- ➗ **Math Solver** — Safely evaluates mathematical expressions.
- 🌍 **Translator** — Translates any text into English using LibreTranslate.
- 🧠 **spaCy-based NER** — Extracts cities and locations accurately from user input.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [OpenAI GPT-4o](https://openai.com/gpt-4o)
- [spaCy](https://spacy.io/)
- [OpenWeather API](https://openweathermap.org/)
- [NewsAPI](https://newsapi.org/)
- [LibreTranslate](https://libretranslate.com/)
- [geopy](https://pypi.org/project/geopy/)
- [timezonefinder](https://pypi.org/project/timezonefinder/)

---

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/intent-chatbot-langgraph.git
cd intent-chatbot-langgraph
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. **Set API keys**
Edit the Python file and replace:
- `openai_api_key` with your OpenAI key
- `your_openweather_api_key` with your OpenWeather key
- `your_newsapi_key_here` with your NewsAPI key

---

## ▶️ Running the Bot

```bash
python chatbot.py
```
Then type your queries in the terminal:
```bash
> What's the weather in Paris?
> Time in Tokyo
> Translate "Bonjour"
> Write a Python loop
> 5 * sin(pi/2)
> Latest news
```
Type `exit` to quit.

---

## 🧪 Example Intents
| Intent     | Example Input                          |
|------------|----------------------------------------|
| `general`  | "How are you today?"                   |
| `weather`  | "What's the weather in London?"        |
| `time`     | "Time in New York"                     |
| `code`     | "Show me a Python function to reverse a string." |
| `news`     | "Give me the latest news"             |
| `math`     | "cos(pi/3) + 5"                         |
| `translate`| "Translate 'Hola' to English"          |

---

## 📌 TODO
- [ ] Add web UI (Streamlit or Flask)
- [ ] Add user-defined language translation support
- [ ] Add news categories (e.g., tech, sports)
- [ ] Store conversation logs

---

## 📄 License
MIT License

---

## 🙌 Acknowledgements
- [LangChain](https://www.langchain.com/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [OpenAI](https://openai.com/)
- [NewsAPI](https://newsapi.org/)
- [OpenWeather](https://openweathermap.org/)
- [LibreTranslate](https://libretranslate.com/)
- [spaCy](https://spacy.io/)

---

Built with ❤️ for learning, fun, and powerful assistants.

