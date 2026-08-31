from hklii_psla.model_factory import get_gpt_model

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage


model = get_gpt_model()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What's the weather in San Francisco?"}
        ]
    }
)

print(result["messages"][-1].content_blocks)
