import os
import logging
import json
from openai import OpenAI
from dotenv import load_dotenv
import requests

# Configure logging to log chatbot activity and errors
logging.basicConfig(filename='chatbot.log', level=logging.INFO)

# Load environment variables from a specified .env file
load_dotenv(r'C:/AI and Machine Learning/api_keys.env')

# Retrieve and verify API keys
OPEN_NEWS_API_KEY = os.getenv("OPEN_NEWS_API_KEY")
if not OPEN_NEWS_API_KEY:
    raise ValueError("No News API key found in C:/AI and Machine Learning/api_keys.env")

OPEN_WEATHER_MAP_API_KEY = os.getenv("OPEN_WEATHER_MAP_API_KEY")
if not OPEN_WEATHER_MAP_API_KEY:
    raise ValueError("No OpenWeatherMap API key found in C:/AI and Machine Learning/api_keys.env")

CAT_API_KEY = os.getenv("CAT_API_KEY")
if not CAT_API_KEY:
    raise ValueError("No Cat API key found in C:/AI and Machine Learning/api_keys.env")

# Initialize OpenAI client with the API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define available tools for the chatbot
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Get the latest news headlines",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The news category (e.g., 'business', 'technology')"
                    }
                },
                "required": ["category"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cat_img",
            "description": "Get a random cat image",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

def get_news(category: str) -> str:
    """Fetch the latest news headlines for a given category."""
    url = f"https://newsapi.org/v2/top-headlines?category={category}&apiKey={OPEN_NEWS_API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    if response.status_code == 200:
        articles = data["articles"]
        headlines = "\n".join([article["title"] for article in articles[:3]])
        return f"🌸 Here are the latest {category} news headlines:\n{headlines}"
    else:
        return f"❌ Error fetching news: {data['message']}"

def get_weather(location: str) -> str:
    """Fetch the current weather for a given location."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPEN_WEATHER_MAP_API_KEY}&units=imperial"
    response = requests.get(url)
    data = response.json()
    
    if response.status_code == 200:
        temp = data["main"]["temp"]
        description = data["weather"][0]["description"]
        return f"🌼 The current weather in {location} is {description} with a temperature of {temp}°F."
    else:
        return f"❌ Error fetching weather: {data['message']}"

def get_cat_img() -> str:
    """Fetch 10 random cat images."""
    url = "https://api.thecatapi.com/v1/images/search?limit=10"
    headers = {"x-api-key": CAT_API_KEY}
    response = requests.get(url, headers=headers)
    data = response.json()
    
    if response.status_code == 200:
        image_urls = [image["url"] for image in data]
        return "🐾 Here are some adorable cat images:\n" + "\n".join(image_urls)
    else:
        return "❌ Error fetching cat images."

def handle_tool_call(tool_call):
    """Handle tool calls based on the API response."""
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    if function_name == "get_news":
        return get_news(**arguments)
    elif function_name == "get_weather":
        return get_weather(**arguments)
    elif function_name == "get_cat_img":
        return get_cat_img()
    else:
        return f"❓ Unknown function: {function_name}"

def chatbot():
    """Main logic for the chatbot interaction."""
    print("👋 Welcome! You can ask for news, weather, or a cat image. Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("👋 Goodbye!")
            break
        
        try:
            # Initial API call with tools
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that can provide news, weather, and cat images."
                    },
                    {"role": "user", "content": user_input}
                ],
                tools=AVAILABLE_TOOLS,
                tool_choice="auto"
            )

            assistant_message = response.choices[0].message
            
            # Handle tool calls if present
            if assistant_message.tool_calls:
                # Collect tool responses
                tool_responses = []
                for tool_call in assistant_message.tool_calls:
                    tool_response = handle_tool_call(tool_call)
                    tool_responses.append({
                        "tool_call_id": tool_call.id,
                        "response": tool_response
                    })
                
                # Prepare messages for second API call for a natural response
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Provide a natural response incorporating the tool results."
                    },
                    {"role": "user", "content": user_input},
                    assistant_message,
                ]
                
                # Add tool responses to messages
                for tool_response in tool_responses:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_response["tool_call_id"],
                        "content": tool_response["response"]
                    })
                
                final_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages
                )
                print(f"🤗 Assistant: {final_response.choices[0].message.content}")
            else:
                print(f"🤗 Assistant: {assistant_message.content}")

        except Exception as e:
            print(f"❗ An error occurred: {str(e)}")
            logging.error(f"Error: {str(e)}")

# Entry point for the chatbot application
if __name__ == "__main__":
    chatbot()
