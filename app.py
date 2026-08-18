import os
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda


# --- 1. Define Tools ---

@tool
def get_country_currency(country: str) -> str:
    """Get the currency used by a given country."""

    currencies = {
        "india": "Indian Rupee (INR)",
        "usa": "United States Dollar (USD)",
        "united states": "United States Dollar (USD)",
        "uk": "British Pound Sterling (GBP)",
        "united kingdom": "British Pound Sterling (GBP)",
        "japan": "Japanese Yen (JPY)",
        "china": "Chinese Yuan (CNY)",
        "australia": "Australian Dollar (AUD)",
        "canada": "Canadian Dollar (CAD)",
        "germany": "Euro (EUR)",
        "france": "Euro (EUR)",
        "italy": "Euro (EUR)",
        "russia": "Russian Ruble (RUB)",
        "south korea": "South Korean Won (KRW)",
        "singapore": "Singapore Dollar (SGD)",
        "uae": "United Arab Emirates Dirham (AED)",
        "saudi arabia": "Saudi Riyal (SAR)",
        "brazil": "Brazilian Real (BRL)",
        "mexico": "Mexican Peso (MXN)",
        "switzerland": "Swiss Franc (CHF)"
    }

    result = currencies.get(country.lower())

    if result:
        return f"The currency of {country.title()} is {result}."

    return f"Currency information not found for {country.title()}."


@tool
def get_country_capital(country: str) -> str:
    """Get the capital city of a given country."""

    capitals = {
        "india": "New Delhi",
        "usa": "Washington, D.C.",
        "united states": "Washington, D.C.",
        "uk": "London",
        "united kingdom": "London",
        "japan": "Tokyo",
        "china": "Beijing",
        "australia": "Canberra",
        "canada": "Ottawa",
        "germany": "Berlin",
        "france": "Paris",
        "italy": "Rome",
        "russia": "Moscow",
        "south korea": "Seoul",
        "singapore": "Singapore",
        "uae": "Abu Dhabi",
        "saudi arabia": "Riyadh",
        "brazil": "Brasilia",
        "mexico": "Mexico City",
        "switzerland": "Bern"
    }

    result = capitals.get(country.lower())

    if result:
        return f"The capital of {country.title()} is {result}."

    return f"Capital information not found for {country.title()}."


tools = [
    get_country_currency,
    get_country_capital
]


# --- 2. Initialize Model & Agent ---

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GOOGLE_API_KEY,
    temperature=0
)


agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are a specialized agent restricted ONLY to country capitals "
        "and country currencies. "
        "You can answer questions about the capital or currency of a country. "
        "For any other topics, questions, or general knowledge outside of "
        "countries, capitals, and currencies, you must say exactly: "
        "'I am not authorized to answer questions outside of countries, capitals, and currencies.'"
    )
)


# --- 3. Input Model ---

class AgentInput(BaseModel):
    input: str = Field(description="Your message to the agent")


def format_for_agent(x) -> dict:
    user_input = x["input"] if isinstance(x, dict) else x.input

    return {
        "messages": [
            ("user", user_input)
        ]
    }


def extract_text_response(agent_output: dict) -> str:

    if not isinstance(agent_output, dict):
        return str(agent_output)

    # Case 1: top-level messages
    messages = agent_output.get("messages")

    # Case 2: nested under a node name
    if messages is None:
        for value in agent_output.values():

            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if messages:

        last = messages[-1]

        return getattr(
            last,
            "content",
            str(last)
        )

    return str(agent_output)


formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# --- 4. FastAPI App ---

app = FastAPI(
    title="Country Information Agent",
    version="1.0",
    description=(
        "A LangChain agent using Gemini with tools for "
        "country capitals and currencies, served via LangServe."
    )
)


@app.get("/")
def root():
    return {
        "message": "Server is running. Visit /agent/playground/ to chat, or /docs for the API."
    }


# --- 5. Add LangServe Route ---

add_routes(
    app,
    formatted_agent_chain,
    path="/agent"
)


# --- 6. Run Server ---

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 8000)
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )