import os
import asyncio

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from huggingface_hub import AsyncInferenceClient

class NaturalLanguagePayload(BaseModel):
    descriptions: list[str]

# Environment Variables
load_dotenv(override=True)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_AUTOREGRESSIVE_MODEL = os.getenv("HF_AUTOREGRESSIVE_MODEL")

completion_client = AsyncInferenceClient(HF_AUTOREGRESSIVE_MODEL, token=HF_API_TOKEN)

app = FastAPI()

# The prompt template may contain parameters for Python string formatting
with open("prompt_template.txt", "r") as template_file:
    prompt_template = template_file.read()

def build_full_prompt(prompt_template, **kwargs):
    return prompt_template.format(**kwargs, length='multi-line')

async def translate_language_to_sql(description: str):
    prompt = build_full_prompt(prompt_template, natural_language_question=description)

    llm_completion = await completion_client.text_generation(prompt, stop=["\n'''"])

    # Remove excessive whitespace characters and trailing apostrophes
    completion_chunks = llm_completion.split()
    reconstructed_completion = " ".join(completion_chunks)
    sql_query, _, _ = reconstructed_completion.partition("'''")
    
    return sql_query.rstrip()

@app.post("/")
async def get_sql_statements(payload: NaturalLanguagePayload):
    descriptions = payload.descriptions

    completion_requests = [translate_language_to_sql(description) for description in descriptions]

    sql_statements = await asyncio.gather(*completion_requests)

    results = [
        {
            "description": description,
            "sql_statement": sql_statement
        }
        for description, sql_statement in zip(descriptions, sql_statements)
    ]

    return { "results": results }
