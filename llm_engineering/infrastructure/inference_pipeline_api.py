import opik
from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from opik import opik_context
from pydantic import BaseModel

from llm_engineering import settings
from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.application.utils import misc
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.infrastructure.opik_utils import configure_opik

configure_opik()

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


@opik.track
def call_llm_service(query: str, context: str | None) -> str:
    if settings.is_offline:
        # Offline: use a local Ollama instance via its OpenAI-compatible API.
        llm = ChatOpenAI(
            model=settings.OLLAMA_MODEL_ID,
            base_url=settings.OLLAMA_BASE_URL,
            api_key="ollama",
            temperature=settings.TEMPERATURE_INFERENCE,
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        response = llm.invoke(messages)
        return response.content

    from llm_engineering.model.inference import InferenceExecutor, LLMInferenceSagemakerEndpoint

    llm = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE, inference_component_name=None
    )
    answer = InferenceExecutor(llm, query, context).execute()

    return answer


@opik.track
def rag(query: str) -> str:
    retriever = ContextRetriever(mock=settings.is_offline)
    documents = retriever.search(query, k=3)
    context = EmbeddedChunk.to_context(documents)

    answer = call_llm_service(query, context)

    if not settings.is_offline:
        try:
            opik_context.update_current_trace(
                tags=["rag"],
                metadata={
                    "model_id": settings.HF_MODEL_ID,
                    "embedding_model_id": settings.TEXT_EMBEDDING_MODEL_ID,
                    "temperature": settings.TEMPERATURE_INFERENCE,
                    "query_tokens": misc.compute_num_tokens(query),
                    "context_tokens": misc.compute_num_tokens(context),
                    "answer_tokens": misc.compute_num_tokens(answer),
                },
            )
        except Exception:
            pass

    return answer


@app.post("/rag", response_model=QueryResponse)
async def rag_endpoint(request: QueryRequest):
    try:
        answer = rag(query=request.query)

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
