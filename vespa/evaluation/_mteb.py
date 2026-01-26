import logging
from typing import Any, Callable, Optional, List, Literal
from pathlib import Path
import hashlib
from vespa.application import Vespa

try:
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
    import simplejson as json
except ImportError as e:
    raise ImportError(
        "The 'mteb' and 'simplejson' package is required for this module but is not installed.\n"
        "Please install them using:\n\n"
        "    pip install pyvespa[mteb]\n\n"
        "or with uv:\n\n"
        "    uv add 'pyvespa[mteb]'"
    ) from e

from vespa.models import ApplicationPackageWithQueryFunctions
from vespa.models import create_hybrid_package, ModelConfig, get_model_config
from vespa.deployment import VespaDocker, VespaCloud
from vespa.application import Vespa as VespaApp
from vespa.io import VespaResponse

# Configure logging with a handler so logs are actually output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class VespaMTEBApp(SearchProtocol):
    """Vespa search using pyvespa with support for Docker and Vespa Cloud deployments."""

    app: Optional[Any] = None
    vespa_docker: Optional[VespaDocker] = None
    vespa_cloud: Optional[VespaCloud] = None
    _fed_indices: set[str] = set()  # Class-level cache of fed indices
    mteb_model_meta: Optional[ModelMeta] = None  # Will be set by MTEB

    def __init__(
        self,
        previous_results: str | None = None,
        port: int = 8080,
        **kwargs,
    ):
        # Get the application package from kwargs (required)
        self.package: ApplicationPackageWithQueryFunctions = kwargs.pop("package")

        # Handle model_config parameter which can be:
        # - A single ModelConfig instance
        # - A string model name
        # - A list of ModelConfig instances or strings
        model_input = kwargs.pop("model_config")

        # Normalize to list of ModelConfig instances for consistent handling
        if isinstance(model_input, str):
            self.model_configs = [get_model_config(model_input)]
        elif isinstance(model_input, ModelConfig):
            self.model_configs = [model_input]
        elif isinstance(model_input, list):
            self.model_configs = [
                get_model_config(m) if isinstance(m, str) else m for m in model_input
            ]
        else:
            raise ValueError(
                f"model_config must be a string, ModelConfig, or list of either. Got {type(model_input)}"
            )

        self.port = port
        self._current_task_name: Optional[str] = None

        # Deployment target configuration
        self.deployment_target: Literal["docker", "cloud"] = kwargs.pop(
            "deployment_target", "cloud"
        )

        # Cloud-specific configuration
        self.tenant: Optional[str] = kwargs.pop("tenant", None)
        self.application: Optional[str] = kwargs.pop("application", None)
        self.instance: str = kwargs.pop("instance", "default")
        self.key_content: Optional[str] = kwargs.pop("key_content", None)
        self.key_location: Optional[str] = kwargs.pop("key_location", None)
        self.auto_cleanup: bool = kwargs.pop("auto_cleanup", True)
        self.disk_folder: Optional[str] = kwargs.pop("disk_folder", None)

        # Validate cloud configuration
        if self.deployment_target == "cloud":
            if self.tenant is None:
                raise ValueError("'tenant' is required when deployment_target='cloud'")
            if self.application is None:
                raise ValueError(
                    "'application' is required when deployment_target='cloud'"
                )

    def cleanup(self) -> None:
        """Stop and remove the Vespa deployment (Docker container or Cloud instance)."""
        if self.deployment_target == "docker":
            # Docker cleanup: stop and remove container
            if (
                self.vespa_docker is not None
                and self.vespa_docker.container is not None
            ):
                container_id = self.vespa_docker.container.short_id
                try:
                    logger.info(f"Stopping Vespa container {container_id}...")
                    self.vespa_docker.container.stop(timeout=10)
                    logger.info(f"Removing Vespa container {container_id}...")
                    self.vespa_docker.container.remove()
                    logger.info(
                        f"Vespa container {container_id} cleaned up successfully."
                    )
                except Exception as e:
                    logger.warning(f"Error during container cleanup: {e}")
                    # Try force removal as fallback
                    try:
                        self.vespa_docker.container.remove(force=True)
                        logger.info(f"Force removed container {container_id}.")
                    except Exception as force_err:
                        logger.error(f"Failed to force remove container: {force_err}")
            self.vespa_docker = None
        else:
            # Cloud cleanup: delete the deployment if auto_cleanup is enabled
            if self.vespa_cloud is not None and self.auto_cleanup:
                try:
                    logger.info(
                        f"Deleting Vespa Cloud deployment (instance={self.instance})..."
                    )
                    self.vespa_cloud.delete(instance=self.instance, environment="dev")
                    logger.info("Vespa Cloud deployment deleted successfully.")
                except Exception as e:
                    logger.warning(f"Error during cloud cleanup: {e}")
            elif self.vespa_cloud is not None:
                logger.info(
                    "auto_cleanup=False, skipping Vespa Cloud deployment deletion."
                )
            self.vespa_cloud = None

        # Clear the fed indices cache since the data is gone
        VespaMTEBApp._fed_indices.clear()
        self.app = None

    def ensure_clean_state(self) -> None:
        """
        Ensure a clean state before starting a new task.

        For Docker: destroys the container and creates a new one.
        For Cloud: deletes all documents but keeps the deployment running (faster).
        """
        if self.deployment_target == "docker":
            # Docker: destroy container and recreate
            if self.app is not None or self.vespa_docker is not None:
                logger.info("Cleaning up existing container before new task...")
                self.cleanup()
        else:
            # Cloud: delete all documents but keep deployment running
            if self.app is not None:
                logger.info("Deleting all documents before new task...")
                try:
                    # Get the content cluster name from the package
                    content_cluster_name = f"{self.package.name}_content"
                    self.app.delete_all_docs(
                        content_cluster_name=content_cluster_name, schema="doc"
                    )
                    logger.info("All documents deleted successfully.")
                    # Clear the fed indices cache since documents are gone
                    VespaMTEBApp._fed_indices.clear()
                except Exception as e:
                    logger.warning(
                        f"Error deleting documents: {e}. Falling back to full cleanup."
                    )
                    self.cleanup()

    def _deploy(self) -> VespaApp:
        """Deploy the Vespa application to Docker or Vespa Cloud."""
        if self.deployment_target == "docker":
            self.vespa_docker = VespaDocker(port=self.port)
            return self.vespa_docker.deploy(self.package)
        else:
            # Cloud deployment
            self.vespa_cloud = VespaCloud(
                tenant=self.tenant,
                application=self.application,
                application_package=self.package,
                key_content=self.key_content,
                key_location=self.key_location,
                instance=self.instance,
            )
            return self.vespa_cloud.deploy(
                instance=self.instance,
                disk_folder=self.disk_folder,
                environment="dev",
            )

    @staticmethod
    def _get_cache_key(model_configs: List[ModelConfig], task_name: str) -> str:
        """Generate a unique cache key for model_configs + task combination."""
        # Create a string representation of all model configs
        config_strs = [f"{mc.model_id}_{mc.embedding_dim}" for mc in model_configs]
        config_str = "_".join(sorted(config_strs))  # Sort for consistency
        combined = f"{config_str}_{task_name}"
        # Use hash for a shorter, consistent key
        return hashlib.md5(combined.encode()).hexdigest()

    def common_query_params(self, top_k: int) -> dict[str, Any]:
        return {
            "hits": top_k,
            "maxHits": top_k,
            "timeout": "20s",
        }

    def is_already_fed(self, task_name: str) -> bool:
        """Check if the index has already been fed for this model config(s) and task."""
        cache_key = self._get_cache_key(self.model_configs, task_name)
        return cache_key in VespaMTEBApp._fed_indices

    def _mark_as_fed(self, task_name: str) -> None:
        """Mark the current model config(s) + task combination as fed."""
        cache_key = self._get_cache_key(self.model_configs, task_name)
        VespaMTEBApp._fed_indices.add(cache_key)
        model_ids = ", ".join([mc.model_id for mc in self.model_configs])
        logger.info(
            f"Marked index as fed for model(s) '{model_ids}' and task '{task_name}'"
        )

    def get_query_functions(self) -> dict[str, Callable]:
        return self.package.get_query_functions()

    def get_timeout(self) -> float:
        return 60.0

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
        num_proc: int = 1,  # Not used, but required by MTEB interface
    ) -> None:
        task_name = task_metadata.name
        self._current_task_name = task_name

        # Get query_function from encode_kwargs
        query_function = encode_kwargs.get("query_function", "default")
        # assert query_function is valid
        if query_function not in self.get_query_functions().keys():
            raise ValueError(
                f"Invalid query_function '{query_function}'. Must be one of {list(self.get_query_functions().keys())}"
            )
        logger.info(f"Using query_function: {query_function}")
        # Check if already fed
        if self.is_already_fed(task_name):
            model_ids = ", ".join([mc.model_id for mc in self.model_configs])
            logger.info(
                f"Index already fed for model(s) '{model_ids}' and task '{task_name}', skipping indexing"
            )
            # Connect to existing application if not already connected
            if self.app is None:
                logger.info("Connecting to existing Vespa application for querying...")
                if self.deployment_target == "docker":
                    self.app = VespaApp(url=f"http://localhost:{self.port}")
                else:
                    # For cloud, use get_application to reconnect
                    if self.vespa_cloud is None:
                        self.vespa_cloud = VespaCloud(
                            tenant=self.tenant,
                            application=self.application,
                            application_package=self.package,
                            key_content=self.key_content,
                            key_location=self.key_location,
                            instance=self.instance,
                        )
                    self.app = self.vespa_cloud.get_application(
                        instance=self.instance,
                        environment="dev",
                        endpoint_type="mtls",
                    )
            return

        model_ids = ", ".join([mc.model_id for mc in self.model_configs])
        logger.info(
            f"Starting indexing for model(s) '{model_ids}' and task '{task_name}'"
        )

        # Ensure clean state before deploying new container
        self.ensure_clean_state()

        logger.info("Deploying Vespa application...")
        self.app: Vespa = self._deploy()

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
                        "text": doc["text"] + doc.get("title", ""),
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

        # Mark as fed after successful indexing
        self._mark_as_fed(task_name)

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
        num_proc: int = 1,  # Not used, but required by MTEB interface
    ) -> RetrievalOutputType:
        if self.app is None:
            raise ValueError("Application not deployed. Call index() first.")

        logger.info(f"Querying Vespa... {len(queries)} queries with top_k={top_k}")

        # Get query function name from encode_kwargs and retrieve the actual function
        query_function_name = encode_kwargs.get("query_function", "default")
        query_functions = self.get_query_functions()
        if query_function_name not in query_functions:
            raise ValueError(
                f"Invalid query_function '{query_function_name}'. Must be one of {list(query_functions.keys())}"
            )
        query_function = query_functions[query_function_name]

        query_ids = list(queries["id"])
        queries_loader = _create_text_queries_dataloader(queries)
        queries_texts = [text for batch in queries_loader for text in batch["text"]]

        # Build query bodies for Vespa
        query_bodies = []
        for query_text in queries_texts:
            query_body = query_function(query_text=query_text, top_k=top_k)
            query_body.update(self.common_query_params(top_k))
            query_bodies.append(query_body)
        # Print sample query body for debugging
        if query_bodies:
            logger.info(f"Sample query body: \n{query_bodies[0]}")
        # Execute queries in parallel
        responses = self.app.query_many(
            query_bodies,
            num_connections=4,
            max_concurrent=10,
            client_kwargs={
                "timeout": self.get_timeout()
            },  # TODO: This can be removed when updating from master
        )

        # Process results
        results = {qid: {} for qid in query_ids}
        empty_results = 0
        for qi, (qid, response) in enumerate(zip(query_ids, responses)):
            if not response.is_successful():
                logger.error(f"Query {qid} failed: {response.status_code}")
                # Add a dummy result with score 0.0 to prevent evaluation pipeline errors
                results[qid] = {"__dummy__": 0.0}
                empty_results += 1
                continue

            hits = response.hits
            doc_id_to_score = {}

            for hit in hits:
                try:
                    doc_id = hit["fields"]["id"]
                    score = hit["relevance"]
                    doc_id_to_score[doc_id] = float(score)
                except (KeyError, TypeError) as e:
                    logger.error(
                        f"Error processing hit for query {qid}: {e}. Hit data: {hit}"
                    )
                    continue

            if len(doc_id_to_score) == 0:
                # Add a dummy result with score 0.0 to prevent evaluation pipeline errors
                doc_id_to_score["__dummy__"] = 0.0
                empty_results += 1
                logger.warning(f"Query {qid} returned no results")

            results[qid] = doc_id_to_score

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


class VespaMTEBEvaluator:
    """
    Evaluator class for running MTEB benchmarks with Vespa.

    This class handles the orchestration of MTEB evaluation tasks using Vespa
    as the search backend. It supports single tasks or full benchmarks, with
    incremental result saving and optional overwrite control.

    Supports both Docker (local) and Vespa Cloud deployments via the
    `deployment_target` parameter.

    Args:
        model_configs: One or more ModelConfig instances or model name strings.
        task_name: Name of a single MTEB task to evaluate (mutually exclusive with benchmark_name).
        benchmark_name: Name of an MTEB benchmark to evaluate (mutually exclusive with task_name).
        results_dir: Directory to save results. Defaults to "results".
        overwrite: If False, skip evaluations where results already exist. Defaults to False.
        deployment_target: Where to deploy Vespa. Either "docker" or "cloud". Defaults to "cloud".
        port: Vespa application port (Docker only). Defaults to 8080.
        tenant: Vespa Cloud tenant name. Required when deployment_target="cloud".
        application: Vespa Cloud application name. Defaults to benchmark/task name if not specified.
        instance: Vespa Cloud instance name. Defaults to "default".
        key_content: Vespa Cloud API key content (string).
        key_location: Path to Vespa Cloud API key file.
        auto_cleanup: Whether to delete cloud deployment after evaluation. Defaults to True.

    Example (Docker):
        >>> evaluator = VespaMTEBEvaluator(
        ...     model_configs="e5-small-v2",
        ...     benchmark_name="NanoBEIR",
        ...     deployment_target="docker",
        ... )
        >>> evaluator.evaluate()

    Example (Vespa Cloud):
        >>> evaluator = VespaMTEBEvaluator(
        ...     model_configs="e5-small-v2",
        ...     benchmark_name="NanoBEIR",
        ...     deployment_target="cloud",
        ...     tenant="my-tenant",
        ...     key_content=os.getenv("VESPA_API_KEY"),
        ... )
        >>> evaluator.evaluate()
    """

    def __init__(
        self,
        model_configs: ModelConfig | str | List[ModelConfig | str],
        task_name: str | None = None,
        benchmark_name: str | None = None,
        results_dir: str | Path = "results",
        overwrite: bool = False,
        deployment_target: Literal["docker", "cloud"] = "cloud",
        port: int = 8080,
        # Cloud-specific parameters
        tenant: str | None = None,
        application: str | None = None,
        instance: str = "default",
        key_content: str | None = None,
        key_location: str | None = None,
        auto_cleanup: bool = True,
    ):
        # Validate mutually exclusive parameters
        if task_name is not None and benchmark_name is not None:
            raise ValueError(
                "Only one of 'task_name' or 'benchmark_name' can be specified, not both."
            )
        if task_name is None and benchmark_name is None:
            raise ValueError(
                "Either 'task_name' or 'benchmark_name' must be specified."
            )

        # Normalize model_configs to list of ModelConfig instances
        self.model_configs = self._normalize_model_configs(model_configs)

        self.task_name = task_name
        self.benchmark_name = benchmark_name
        self.results_dir = Path(results_dir)
        self.overwrite = overwrite
        self.port = port

        # Deployment configuration
        self.deployment_target = deployment_target
        self.tenant = tenant
        self.instance = instance
        self.key_content = key_content
        self.key_location = key_location
        self.auto_cleanup = auto_cleanup

        # Derive application name from benchmark/task name if not specified
        if application is None:
            eval_name = benchmark_name or task_name
            # Sanitize the name for Vespa Cloud (lowercase, alphanumeric and hyphens)
            self.application = eval_name.lower().replace("_", "-")
        else:
            self.application = application

        # Validate cloud configuration
        if deployment_target == "cloud":
            if tenant is None:
                raise ValueError("'tenant' is required when deployment_target='cloud'")

        # Create the application package
        self.package = create_hybrid_package(self.model_configs)
        self.query_function_names = list(self.package.get_query_functions().keys())

        # Compute model suffix for file naming
        self.model_suffix = self._get_model_suffix()

        # Track the VespaMTEBApp instance for lifecycle management
        self._vespa_app: Optional[VespaMTEBApp] = None

    def _normalize_model_configs(
        self, model_input: ModelConfig | str | List[ModelConfig | str]
    ) -> List[ModelConfig]:
        """Normalize model_config input to a list of ModelConfig instances."""
        if isinstance(model_input, str):
            return [get_model_config(model_input)]
        elif isinstance(model_input, ModelConfig):
            return [model_input]
        elif isinstance(model_input, list):
            return [
                get_model_config(m) if isinstance(m, str) else m for m in model_input
            ]
        else:
            raise ValueError(
                f"model_configs must be a string, ModelConfig, or list of either. Got {type(model_input)}"
            )

    def _get_model_suffix(self) -> str:
        """
        Generate a suffix string for file/directory naming based on model configs.

        Uses only unique model IDs to keep filenames short.
        For single model: {model_id}
        For multiple models: {model1_id}__{model2_id}
        """
        # Use a set to get unique model IDs, then sort for consistent ordering
        unique_model_ids = sorted(set(config.model_id for config in self.model_configs))
        return "__".join(unique_model_ids)

    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()

    def _load_benchmark_results(self, results_path: Path) -> dict:
        """Load existing benchmark results from file, or return empty structure."""
        if results_path.exists():
            with open(results_path, "r") as f:
                return json.load(f)
        return {}

    def _save_benchmark_results(self, results_path: Path, results: dict) -> None:
        """Save benchmark results to file with atomic write."""
        # Write to temp file first, then rename for atomic operation
        temp_path = results_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(results, f, indent=4, ignore_nan=True)
        temp_path.rename(results_path)

    def _is_task_query_function_complete(
        self, benchmark_results: dict, task_name: str, query_function: str
    ) -> bool:
        """Check if a specific task + query_function combination has already been evaluated."""
        return (
            "results" in benchmark_results
            and task_name in benchmark_results["results"]
            and query_function in benchmark_results["results"][task_name]
            and benchmark_results["results"][task_name][query_function].get(
                "finished_at"
            )
            is not None
        )

    def _get_results_path(self, eval_name: str) -> Path:
        """Get the path for the results file."""
        results_dir = self.results_dir / eval_name
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = f"{eval_name}_{self.model_suffix}_results.json"
        return results_dir / results_file

    def _save_package(self, eval_name: str) -> Path:
        """Save the application package to disk."""
        package_dir = self.results_dir / "packages" / eval_name / self.model_suffix
        package_dir.parent.mkdir(parents=True, exist_ok=True)
        self.package.to_files(package_dir)
        return package_dir

    def cleanup(self) -> None:
        """Clean up the Vespa app instance if one exists."""
        if self._vespa_app is not None:
            self._vespa_app.cleanup()
            self._vespa_app = None

    def _create_loader(self):
        """
        Create a loader function that captures this evaluator instance.

        The loader reuses the existing VespaMTEBApp instance if available,
        or creates a new one. This avoids global state while allowing
        MTEB to instantiate the model via its standard loader pattern.
        """

        def loader(model_name, **kwargs) -> SearchProtocol:
            if self._vespa_app is not None:
                return self._vespa_app
            self._vespa_app = VespaMTEBApp(**kwargs)
            return self._vespa_app

        return loader

    def get_model_meta(self) -> ModelMeta:
        """
        Get the MTEB ModelMeta for this evaluator's configuration.

        Returns:
            ModelMeta instance configured for Vespa search.
        """
        # Prepare disk folder for cloud config files
        disk_folder = str(self.results_dir / "cloud_config" / self.application)

        return ModelMeta(
            loader=self._create_loader(),
            loader_kwargs={
                "model_config": self.model_configs,
                "package": self.package,
                "port": self.port,
                "deployment_target": self.deployment_target,
                "tenant": self.tenant,
                "application": self.application,
                "instance": self.instance,
                "key_content": self.key_content,
                "key_location": self.key_location,
                "auto_cleanup": self.auto_cleanup,
                "disk_folder": disk_folder,
            },
            name=f"vespa/{self.model_suffix}_hybrid_vespa",
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

    def _get_tasks(self) -> list:
        """Get the list of tasks to evaluate."""
        if self.benchmark_name is not None:
            benchmark = mteb.get_benchmark(self.benchmark_name)
            return list(benchmark.tasks)
        # task_name is guaranteed to be set by __init__ validation
        return [mteb.get_task(self.task_name)]  # type: ignore[arg-type]

    def _get_eval_name(self) -> str:
        """Get the evaluation name (benchmark or task name)."""
        # One of these is guaranteed to be set by __init__ validation
        return self.benchmark_name or self.task_name  # type: ignore[return-value]

    def evaluate(self) -> dict:
        """
        Run the MTEB evaluation.

        Returns:
            dict: The benchmark results including metadata and scores for all
                  task/query_function combinations.
        """
        eval_name = self._get_eval_name()
        tasks = self._get_tasks()

        # Setup results path and check for existing results
        results_path = self._get_results_path(eval_name)

        # Check if results file exists and overwrite is False
        if results_path.exists() and not self.overwrite:
            existing_results = self._load_benchmark_results(results_path)
            # Check if benchmark is already complete
            if (
                existing_results.get("metadata", {}).get("benchmark_finished_at")
                is not None
            ):
                logger.info(
                    f"Results file already exists and benchmark is complete: {results_path}\n"
                    f"Skipping evaluation. Set overwrite=True to re-run."
                )
                return existing_results

        # Save the application package
        package_dir = self._save_package(eval_name)
        logger.info(f"Saved application package to: {package_dir}")
        logger.info(f"Available query functions: {self.query_function_names}")

        # Load existing results (for incremental saving) or start fresh if overwrite=True
        if self.overwrite:
            benchmark_results = {}
        else:
            benchmark_results = self._load_benchmark_results(results_path)

        # Initialize metadata if not present
        if "metadata" not in benchmark_results:
            benchmark_results["metadata"] = {
                "benchmark_name": eval_name,
                "model_configs": [config.to_dict() for config in self.model_configs],
                "query_functions": self.query_function_names,
                "tasks": [task.metadata.name for task in tasks],
                "benchmark_started_at": self._get_timestamp(),
                "benchmark_finished_at": None,
            }

        # Initialize results structure if not present
        if "results" not in benchmark_results:
            benchmark_results["results"] = {}

        # Save initial state
        self._save_benchmark_results(results_path, benchmark_results)

        try:
            vespa_model_meta = self.get_model_meta()

            for task in tasks:
                task_name = task.metadata.name
                logger.info(f"Starting evaluation for task: {task_name}")

                # Initialize task results if not present
                if task_name not in benchmark_results["results"]:
                    benchmark_results["results"][task_name] = {}

                for query_function in self.query_function_names:
                    # Check if this task + query_function is already complete
                    if not self.overwrite and self._is_task_query_function_complete(
                        benchmark_results, task_name, query_function
                    ):
                        logger.info(
                            f"Task '{task_name}' with query function '{query_function}' already complete, skipping. "
                            f"Set overwrite=True to re-run."
                        )
                        continue

                    logger.info(
                        f"Evaluating task '{task_name}' with query function: {query_function}"
                    )

                    # Record start time
                    started_at = self._get_timestamp()

                    # Initialize entry for this query function
                    benchmark_results["results"][task_name][query_function] = {
                        "started_at": started_at,
                        "finished_at": None,
                        "scores": None,
                    }
                    # Save incremental progress (started)
                    self._save_benchmark_results(results_path, benchmark_results)

                    # Get a fresh task instance to avoid state corruption between evaluations
                    # MTEB tasks can have internal state (like dataset) that gets corrupted
                    # when reused across multiple evaluate() calls
                    fresh_task = mteb.get_task(task_name)

                    results = mteb.evaluate(
                        vespa_model_meta,
                        fresh_task,
                        encode_kwargs={"query_function": query_function},
                        overwrite_strategy="always",
                    )

                    # Record finish time and scores
                    finished_at = self._get_timestamp()
                    benchmark_results["results"][task_name][query_function] = {
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "scores": results[0].scores,
                    }

                    # Save incremental progress (finished)
                    self._save_benchmark_results(results_path, benchmark_results)

                    # Print results summary
                    self._print_results_summary(
                        task_name, query_function, results[0].scores
                    )

                # Note: Container cleanup happens automatically in index() via
                # ensure_clean_state() before each new task, not after.
                # This avoids interrupting any pending queries.

            # Mark benchmark as complete
            benchmark_results["metadata"]["benchmark_finished_at"] = (
                self._get_timestamp()
            )
            self._save_benchmark_results(results_path, benchmark_results)
            logger.info(f"Evaluation complete! Results saved to: {results_path}")

        finally:
            # Ensure cleanup even if an error occurred
            self.cleanup()
            logger.info("Evaluation finished.")

        return benchmark_results

    def _print_results_summary(
        self, task_name: str, query_function: str, scores: dict
    ) -> None:
        """Print NDCG@10 for the evaluation results."""
        print(f"Finished evaluation for '{task_name}' with '{query_function}':")
        if "train" in scores and "ndcg_at_10" in scores["train"][0]:
            print(f"  NDCG@10: {scores['train'][0]['ndcg_at_10']}")
        elif "test" in scores and "ndcg_at_10" in scores["test"][0]:
            print(f"  NDCG@10: {scores['test'][0]['ndcg_at_10']}")
        else:
            print(f"  Scores: {scores}")
