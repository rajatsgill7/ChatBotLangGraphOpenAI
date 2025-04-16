from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, LLMChain
from typing import TypedDict, Literal, Annotated
from datetime import datetime
from timezonefinder import TimezoneFinder
from geopy.geocoders import Nominatim
import pytz
import requests
import re
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# 1. Define the shared state
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: Literal["general", "weather", "code", "time", "news", "math", "translate"] | None

# 2. Initialize the model and shared memory
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    openai_api_key=""
)

memory = ConversationBufferMemory(return_messages=True)
general_chat_chain = ConversationChain(llm=llm, memory=memory)

weather_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a weather assistant. Answer weather-related questions based on provided location."),
    ("user", "{input}")
])
weather_chain = LLMChain(llm=llm, prompt=weather_prompt, memory=memory)

code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a coding assistant. Generate clean, commented code."),
    ("user", "{input}")
])
code_chain = LLMChain(llm=llm, prompt=code_prompt, memory=memory)

time_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world clock assistant. Help users find the time in various cities."),
    ("user", "{input}")
])
time_chain = LLMChain(llm=llm, prompt=time_prompt, memory=memory)

# Helper function using spaCy NER
def detect_city(text: str) -> str | None:
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            return ent.text
    return None

# 3. Define nodes

def classify_intent(state: GraphState):
    last_msg = state["messages"][-1].content
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify the user's intent into one of: general, weather, code, time, news, math, translate."),
        ("user", "{input}")
    ])
    chain = prompt | llm
    intent = chain.invoke({"input": last_msg}).content.strip().lower()

    valid_intents = {"general", "weather", "code", "time", "news", "math", "translate"}
    if intent not in valid_intents:
        intent = "general"
    return {"intent": intent}

def general_chat(state: GraphState):
    last_msg = state["messages"][-1].content
    output = general_chat_chain.predict(input=last_msg)
    return {"messages": [{"role": "assistant", "content": output}]}

def weather_node(state: GraphState):
    last_msg = state["messages"][-1].content
    location = detect_city(last_msg)
    if not location:
        return {"messages": [{"role": "assistant", "content": "Please specify a city for the weather lookup."}]}

    api_key = ""
    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    response = requests.get(url).json()
    if response.get("cod") != 200:
        return {"messages": [{"role": "assistant", "content": f"Couldn't fetch weather for '{location}': {response.get('message')}"}]}

    temp = response["main"]["temp"]
    desc = response["weather"][0]["description"]
    summary = f"It's currently {temp}°C with {desc} in {location.title()}."
    memory.chat_memory.add_user_message(last_msg)
    memory.chat_memory.add_ai_message(summary)
    return {"messages": [{"role": "assistant", "content": summary}]}

def code_node(state: GraphState):
    last_msg = state["messages"][-1].content
    output = code_chain.run(input=last_msg)
    return {"messages": [{"role": "assistant", "content": output}]}

def time_node(state: GraphState):
    last_msg = state["messages"][-1].content
    location = detect_city(last_msg)
    if not location:
        return {"messages": [{"role": "assistant", "content": "Please specify a city for the time lookup."}]}

    geolocator = Nominatim(user_agent="langgraph-bot")
    location_data = geolocator.geocode(location)
    if not location_data:
        return {"messages": [{"role": "assistant", "content": f"Couldn't locate the city '{location}'."}]}
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lng=location_data.longitude, lat=location_data.latitude)
    if not tz_name:
        return {"messages": [{"role": "assistant", "content": f"Couldn't determine timezone for '{location}'."}]}
    tz = pytz.timezone(tz_name)
    now = datetime.now(tz).strftime("%A, %d %B %Y %I:%M %p %Z")
    summary = f"The current time in {location.title()} is {now}."
    memory.chat_memory.add_user_message(last_msg)
    memory.chat_memory.add_ai_message(summary)
    return {"messages": [{"role": "assistant", "content": summary}]}

def news_node(state: GraphState):
    api_key = ""
    url = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=5&apiKey={api_key}"
    response = requests.get(url).json()
    if response.get("status") != "ok":
        return {"messages": [{"role": "assistant", "content": "Couldn't fetch news at the moment."}]}

    articles = response.get("articles", [])
    if not articles:
        return {"messages": [{"role": "assistant", "content": "No top news found."}]}

    headlines = "\n".join([f"{idx + 1}. {article['title']}" for idx, article in enumerate(articles)])
    return {"messages": [{"role": "assistant", "content": f"📰 Top News Headlines:\n\n{headlines}"}]}

def math_node(state: GraphState):
    import math
    last_msg = state["messages"][-1].content
    try:
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(last_msg, {"__builtins__": None}, allowed_names)
        return {"messages": [{"role": "assistant", "content": f"The result is: {result}"}]}
    except Exception as e:
        return {"messages": [{"role": "assistant", "content": f"Invalid math expression: {str(e)}"}]}

def translate_node(state: GraphState):
    last_msg = state["messages"][-1].content
    payload = {
        "q": last_msg,
        "source": "auto",
        "target": "en",
        "format": "text"
    }
    try:
        response = requests.post("https://libretranslate.com/translate", json=payload)
        data = response.json()
        translation = data.get("translatedText")
        if translation:
            return {"messages": [{"role": "assistant", "content": f"Translated to English:\n\n{translation}"}]}
        else:
            return {"messages": [{"role": "assistant", "content": "Sorry, I couldn't translate that."}]}
    except Exception as e:
        return {"messages": [{"role": "assistant", "content": f"Error translating: {str(e)}"}]}

# 4. Build the graph
graph = StateGraph(GraphState)
graph.add_node("classify_intent", classify_intent)
graph.add_node("general", general_chat)
graph.add_node("weather", weather_node)
graph.add_node("code", code_node)
graph.add_node("time", time_node)
graph.add_node("news", news_node)
graph.add_node("math", math_node)
graph.add_node("translate", translate_node)

graph.set_entry_point("classify_intent")
graph.add_conditional_edges("classify_intent", lambda state: state["intent"], {
    "general": "general",
    "weather": "weather",
    "code": "code",
    "time": "time",
    "news": "news",
    "math": "math",
    "translate": "translate"
})

for node in ["general", "weather", "code", "time", "news", "math", "translate"]:
    graph.add_edge(node, END)

chat_bot = graph.compile()

# 5. Run the bot
if __name__ == "__main__":
    print("ChatBot ready. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        result = chat_bot.invoke({"messages": [{"role": "user", "content": user_input}], "intent": None})
        print("Bot:", result["messages"][-1].content)