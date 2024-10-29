# Natural Language to SQL API

This API translates natural language descriptions into SQL statements for an existing dataset. Built on FastAPI in Python 3.11, it is licensed under a GNU Affero General Public License (AGPLv3).

## 1. Project Setup

For machine learning worloads, the API relies on the [HuggingFace Inference API](https://huggingface.co/docs/huggingface_hub/guides/inference). The following environment variables must be set in the respective `.env` file.

* `HG_AUTOREGRESSIVE_MODEL` - A HuggingFace autoregressive large language model (LLM) for text generation. For instance: `meta-llama/Llama-3.2-1B` (used during development).
* `HG_API_TOKEN` - A HuggingFace API token.

To install Python dependencies, simply run:

```sh
pip install -r requirements.txt
```

## 2. Development Enviroment

To start the development server with hot-reloading abilities, run:

```sh
fastapi dev main.py
```

## 3. Usage and Endpoints

The main endpoint is available at the root of the service (i.e., `/`) for `POST` requests. It takes one or more natural language descriptions to be converted into SQL statements. The body of each request must be of the following form:

```json
{
    "descriptions": [
        "Find total contributions by each contributor.",
        "List all contributions made in the year 2022."
    ]
}
```

For more information see, the [API docs on the local development server](http:localhost:8000/docs).

### 4. Deployment

To use in production, run:

```sh
fastapi run main.py
```