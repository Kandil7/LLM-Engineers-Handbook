import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger
from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering import settings
from llm_engineering.application import utils
from llm_engineering.domain.base.nosql import NoSQLBaseDocument
from llm_engineering.domain.documents import ArticleDocument, Document, PostDocument, RepositoryDocument, UserDocument


@step
def query_data_warehouse(
    author_full_names: list[str],
) -> Annotated[list, "raw_documents"]:
    if settings.is_offline:
        logger.info("OFFLINE_MODE active: loading raw documents from bundled JSON files.")
        documents = _load_from_bundled_json()
        step_context = get_step_context()
        step_context.add_output_metadata(output_name="raw_documents", metadata=_get_metadata(documents))
        return documents

    documents = []
    authors = []
    for author_full_name in author_full_names:
        logger.info(f"Querying data warehouse for user: {author_full_name}")

        first_name, last_name = utils.split_user_full_name(author_full_name)
        logger.info(f"First name: {first_name}, Last name: {last_name}")
        user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)
        authors.append(user)

        results = fetch_all_data(user)
        user_documents = [doc for query_result in results.values() for doc in query_result]

        documents.extend(user_documents)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="raw_documents", metadata=_get_metadata(documents))

    return documents


def _load_from_bundled_json() -> list:
    data_dir = Path("data/data_warehouse_raw_data")
    # UserDocument is only needed for author lookup in the online path; skip it
    # in offline mode so the cleaning/embedding pipeline never sees "users".
    category_classes = {
        "ArticleDocument": ArticleDocument,
        "PostDocument": PostDocument,
        "RepositoryDocument": RepositoryDocument,
    }
    documents = []
    if not data_dir.is_dir():
        logger.warning(f"Bundled data dir not found: {data_dir}")
        return documents
    for file in data_dir.iterdir():
        if not file.is_file():
            continue
        category_class = category_classes.get(file.stem)
        if not category_class:
            continue
        try:
            with file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                try:
                    documents.append(category_class.from_mongo(d))
                except Exception as e:
                    logger.warning(f"Skipping a document from {file.name}: {e}")
        except Exception as e:
            logger.exception(f"Failed to load {file}: {e}")
    logger.info(f"Loaded {len(documents)} documents from bundled JSON.")
    return documents


def fetch_all_data(user: UserDocument) -> dict[str, list[NoSQLBaseDocument]]:
    user_id = str(user.id)
    with ThreadPoolExecutor() as executor:
        future_to_query = {
            executor.submit(__fetch_articles, user_id): "articles",
            executor.submit(__fetch_posts, user_id): "posts",
            executor.submit(__fetch_repositories, user_id): "repositories",
        }

        results = {}
        for future in as_completed(future_to_query):
            query_name = future_to_query[future]
            try:
                results[query_name] = future.result()
            except Exception:
                logger.exception(f"'{query_name}' request failed.")

                results[query_name] = []

    return results


def __fetch_articles(user_id) -> list[NoSQLBaseDocument]:
    return ArticleDocument.bulk_find(author_id=user_id)


def __fetch_posts(user_id) -> list[NoSQLBaseDocument]:
    return PostDocument.bulk_find(author_id=user_id)


def __fetch_repositories(user_id) -> list[NoSQLBaseDocument]:
    return RepositoryDocument.bulk_find(author_id=user_id)


def _get_metadata(documents: list[Document]) -> dict:
    metadata = {
        "num_documents": len(documents),
    }
    for document in documents:
        collection = document.get_collection_name()
        if collection not in metadata:
            metadata[collection] = {}
        if "authors" not in metadata[collection]:
            metadata[collection]["authors"] = list()

        metadata[collection]["num_documents"] = metadata[collection].get("num_documents", 0) + 1
        author = getattr(document, "author_full_name", None)
        if author is not None:
            metadata[collection]["authors"].append(author)

    for value in metadata.values():
        if isinstance(value, dict) and "authors" in value:
            value["authors"] = list(set(value["authors"]))

    return metadata
