
import os

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


# Step 1: Product database
products = {
    "wireless headphones": {
        "price": 79.99,
        "rating": 4.6,
        "description": "Over-ear Bluetooth, 30-hour battery, active noise cancellation"
    },

    "smart watch": {
        "price": 199.99,
        "rating": 4.3,
        "description": "Tracks heart rate and sleep, 5-day battery, water-resistant"
    },

    "mechanical keyboard": {
        "price": 129.99,
        "rating": 4.8,
        "description": "Tenkeyless, Cherry MX Brown switches"
    },

    "laptop stand": {
        "price": 34.99,
        "rating": 4.5,
        "description": "Adjustable aluminium, fits laptops up to 18 inches"
    }
}


# Step 2: Product review database
reviews = {
    "wireless headphones": {
        "reviews": 1234,
        "rating": 4.6
    },

    "smart watch": {
        "reviews": 340,
        "rating": 3.9
    },

    "mechanical keyboard": {
        "reviews": 345,
        "rating": 4.5
    },

    "laptop stand": {
        "reviews": 567,
        "rating": 3.4
    }
}


# Step 3: Create product lookup tool
@tool
def get_product(name: str) -> str:
    """Look up a product by name and return its price, description and rating."""

    product = products.get(name.lower().strip())

    if not product:
        return f"Product not found. Available: {', '.join(products.keys())}"

    return str(product)


# Step 4: Create review lookup tool
@tool
def get_review(name: str) -> str:
    """Look up a product by name and return its number of reviews and rating."""

    review = reviews.get(name.lower().strip())

    if not review:
        return "Reviews for this product were not found."

    return str(review)


# Step 5: Configure Groq API
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not configured.")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0
)


# Step 6: Create AI agent
system_prompt = """
You are a helpful product assistant for an online tech store.

Use the available tools to answer questions about product prices,
descriptions, ratings and customer reviews.

Do not invent product information.
"""

checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=[get_product, get_review],
    system_prompt=system_prompt,
    checkpointer=checkpointer
)


# Step 7: Function to ask questions
def ask(question: str, thread_id: str = "user-session-1"):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        },
        config=config
    )

    print("\nAI Assistant:")
    print(result["messages"][-1].content)


# Step 8: Test the agent
ask("What is the price of wireless headphones")

ask("How do people like the smart watch?")

ask("What is the price and reviews of smart watch?")

ask("What is the price of mechanical keyboard?")

ask("What about its ratings?")
