import chainlit as cl
from agents import Agent, Runner, RunConfig, OpenAIChatCompletionsModel, AsyncOpenAI
from dotenv import load_dotenv, find_dotenv
import os
# Load environment variables from .env file
load_dotenv(find_dotenv())


external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)


agent: Agent = Agent(
    name="Gemini Agent",
    model=model,
    instructions="You are a helpful assistant that answers questions.",
)


@cl.on_message
async def on_message(message: cl.Message):
    result = Runner.run_sync(
        agent,
        input=message.content,
        run_config=config
    )
    await cl.Message(result.final_output).send()