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

prompt_template = """
# Task 
Generate a SQL query to answer the following question: `{natural_language_question}`.

It may be necessary to infer additional information from the question such as the names of individuals, places,
or activities.

## Context
The dataset contains information from Alameda Research and FTX about contributions from different contributors
to different recipients. The data includes details such as contribution amount, contributor information,
recipient information, dates, and states. Alameda Research and FTX were both founded by Sam Bankman-Fried
(colloquially known as SBF).

### PostgreSQL Database Schema 
The query will run on a database with the following schema: 

CREATE TABLE contributions (
    id SERIAL PRIMARY KEY,  -- Unique identifier for each record
    cycle INT NOT NULL,  -- Election cycle
    state_federal VARCHAR(10) NOT NULL,  -- State or Federal
    contribid VARCHAR(20) NOT NULL,  -- Contributor ID
    contrib VARCHAR(100) NOT NULL,  -- Contributor name
    city VARCHAR(100) NOT NULL,  -- City
    state CHAR(2) NOT NULL,  -- State abbreviation
    zip VARCHAR(10),  -- Zip code
    fecoccemp VARCHAR(100),  -- Occupation/Employer
    orgname VARCHAR(100),  -- Organization name
    ultorg VARCHAR(100),  -- Ultimate organization
    date DATE NOT NULL,  -- Date of contribution
    amount DECIMAL(10, 2) NOT NULL,  -- Amount contributed
    recipid VARCHAR(20) NOT NULL,  -- Recipient ID
    recipient VARCHAR(100) NOT NULL,  -- Recipient name
    party CHAR(1) NOT NULL,  -- Party affiliation (D, R, etc.)
    recipcode VARCHAR(10),  -- Recipient code
    type VARCHAR(10),  -- Type of contribution
    fectransid VARCHAR(20),  -- FEC transaction ID
    pg VARCHAR(10),  -- Page number
    cmteid VARCHAR(20)  -- Committee ID
);

# SQL 
Here is the SQL query that answers the question: `{natural_language_question}`.
\'''sql
"""

async def translate_language_to_sql(description: str):
    prompt = prompt_template.format(natural_language_question=description, length='multi-line')

    llm_completion = await completion_client.text_generation(prompt, stop=["'''"])

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
