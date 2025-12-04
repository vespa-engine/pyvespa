# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mteb",
#     "pyvespa",
# ]
# ///
import logging
from typing import Any, Optional

import mteb
from mteb._create_dataloaders import _create_text_queries_dataloader
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.model_meta import ModelMeta
from mteb.models.models_protocols import SearchProtocol
from mteb.types import (
    CorpusDatasetType,
    InstructionDatasetType,
    QueryDatasetType,
    RetrievalOutputType,
    TopRankedDocumentsType,
)

logger = logging.getLogger(__name__)


def vespa_loader(model_name, **kwargs) -> SearchProtocol:
    # requires_package(vespa_loader, "pyvespa", model_name, "pip install pyvespa")
    from vespa.package import (
        ApplicationPackage,
        Field,
        Schema,
        Document,
        RankProfile,
        FieldSet,
        Function,
    )
    from vespa.deployment import VespaDocker
    from vespa.io import VespaResponse
    import vespa.querybuilder as qb

    class VespaSearchApp:
        """Vespa search app using pyvespa"""

        app: Optional[Any] = None
        vespa_docker: Optional[VespaDocker] = None

        def __init__(
            self,
            previous_results: str | None = None,
            port: int = 8080,
            **kwargs,
        ):
            self.port = port
            self.package = ApplicationPackage(
                name="mtebbm25",
                schema=[
                    Schema(
                        name="doc",
                        document=Document(
                            fields=[
                                Field(
                                    name="id",
                                    type="string",
                                    indexing=["summary", "attribute"],
                                ),
                                Field(
                                    name="title",
                                    type="string",
                                    indexing=["index", "summary"],
                                    index="enable-bm25",
                                ),
                                Field(
                                    name="text",
                                    type="string",
                                    indexing=["index", "summary"],
                                    index="enable-bm25",
                                    bolding=True,
                                ),
                            ]
                        ),
                        fieldsets=[FieldSet(name="default", fields=["title", "text"])],
                        rank_profiles=[
                            RankProfile(
                                name="bm25",
                                functions=[
                                    Function(
                                        name="bm25sum",
                                        expression="bm25(title) + bm25(text)",
                                    )
                                ],
                                first_phase="bm25sum",
                            )
                        ],
                    ),
                ],
            )

        def index(
            self,
            corpus: CorpusDatasetType,
            *,
            task_metadata: TaskMetadata,
            hf_split: str,
            hf_subset: str,
            encode_kwargs: dict[str, Any],
        ) -> None:
            logger.info("Deploying Vespa application...")
            # Deploy Vespa application
            self.vespa_docker = VespaDocker(port=self.port)
            self.app = self.vespa_docker.deploy(application_package=self.package)

            logger.info("Starting to index corpus...")

            # Convert corpus to Vespa feed format and count documents
            doc_count = 0
            fed_count = 0
            error_count = 0

            def corpus_to_vespa_feed(corpus):
                nonlocal doc_count
                for doc in corpus:
                    doc_count += 1
                    yield {
                        "id": doc["id"],
                        "fields": {
                            "id": doc["id"],
                            "title": doc.get("title", ""),
                            "text": doc["text"],
                        },
                    }

            # Feed documents to Vespa
            def feed_callback(response: VespaResponse, doc_id: str):
                nonlocal fed_count, error_count
                if response.is_successful():
                    fed_count += 1
                    if fed_count % 100 == 0:
                        logger.info(f"Fed {fed_count} documents...")
                else:
                    error_count += 1
                    logger.error(f"Error feeding doc {doc_id}: {response.json}")

            vespa_feed = corpus_to_vespa_feed(corpus)
            self.app.feed_iterable(vespa_feed, schema="doc", callback=feed_callback)

            logger.info(
                f"Successfully indexed {fed_count} documents to Vespa (errors: {error_count})"
            )

        def search(
            self,
            queries: QueryDatasetType,
            *,
            task_metadata: TaskMetadata,
            hf_split: str,
            hf_subset: str,
            top_k: int,
            encode_kwargs: dict[str, Any],
            instructions: InstructionDatasetType | None = None,
            top_ranked: TopRankedDocumentsType | None = None,
        ) -> RetrievalOutputType:
            if self.app is None:
                raise ValueError("Application not deployed. Call index() first.")

            logger.info(f"Querying Vespa... {len(queries)} queries with top_k={top_k}")

            query_ids = list(queries["id"])
            queries_loader = _create_text_queries_dataloader(queries)
            queries_texts = [text for batch in queries_loader for text in batch["text"]]

            # Build query bodies for Vespa
            query_bodies = []
            for query_text in queries_texts:
                query_body = {
                    "yql": str(
                        qb.select("*")
                        .from_("sources *")
                        .where(qb.userQuery())
                        .set_limit(top_k)
                    ),
                    "query": query_text,
                    "ranking": "bm25",
                    "hits": top_k,
                    "maxHits": top_k,
                }
                query_bodies.append(query_body)

            # Execute queries in parallel
            responses = self.app.query_many(
                query_bodies,
                num_connections=1,
                max_concurrent=10,
            )

            # Process results
            results = {qid: {} for qid in query_ids}
            empty_results = 0
            for qi, (qid, response) in enumerate(zip(query_ids, responses)):
                if not response.is_successful():
                    logger.error(f"Query {qid} failed: {response.status_code}")
                    # Still include empty dict for failed queries
                    results[qid] = {}
                    empty_results += 1
                    continue

                hits = response.hits
                doc_id_to_score = {}

                for hit in hits:
                    doc_id = hit["fields"]["id"]
                    score = hit["relevance"]
                    doc_id_to_score[doc_id] = float(score)

                results[qid] = doc_id_to_score

                if len(doc_id_to_score) == 0:
                    empty_results += 1
                    logger.warning(f"Query {qid} returned no results")

            logger.info(
                f"Completed queries: {len(results)} total, {empty_results} with no results"
            )

            # Debug: print sample results
            if query_ids:
                sample_qid = query_ids[0]
                sample_results = results[sample_qid]
                logger.info(f"Sample query {sample_qid}: {len(sample_results)} results")
                if sample_results:
                    logger.info(f"Sample scores: {list(sample_results.values())[:3]}")

            return results

        def __del__(self):
            """Clean up Vespa container when object is destroyed."""
            if self.vespa_docker is not None:
                try:
                    logger.info("Cleaning up Vespa container...")
                    self.vespa_docker.container.stop(timeout=10)
                    self.vespa_docker.container.remove()
                except Exception as e:
                    logger.warning(f"Error during cleanup: {e}")

    return VespaSearchApp(**kwargs)


vespa_bm25 = ModelMeta(
    loader=vespa_loader,
    loader_kwargs={},
    name="vespa/bm25",
    languages=["eng-Latn"],
    open_weights=True,
    revision="1.0.0",
    release_date="2024-12-04",
    n_parameters=None,
    memory_usage_mb=None,
    embed_dim=None,
    license="apache-2.0",
    max_tokens=None,
    reference="https://docs.vespa.ai/en/reference/bm25.html",
    similarity_fn_name=None,
    framework=[],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    citation="""@article{vespa,
      title={Vespa: The Open Big Data Serving Engine},
      author={Vespa.ai},
      year={2024},
      url={https://vespa.ai},
}""",
)

if __name__ == "__main__":
    task = mteb.get_task("NanoMSMARCORetrieval")
    results = mteb.evaluate(vespa_bm25, task, overwrite_strategy="always")
    print("Finished evaluation, results:")
    print(f"NDCG@10: {results[0].scores['train'][0]['ndcg_at_10']}")
    print("Cleaning up Vespa container...")
    del vespa_bm25
