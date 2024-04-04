import asyncio
import aiohttp
import ujson
import requests
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential
from rich.progress import Progress, BarColumn, TextColumn, TransferSpeedColumn
from rich.console import Console


class VespaFeeder:
    def __init__(self, endpoint, documents, max_connections=100, max_workers=10):
        self.endpoint = endpoint
        self.documents = documents
        self.max_connections = max_connections
        self.max_workers = max_workers
        self.session = requests.Session()
        self.async_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=self.max_connections),
            json_serialize=ujson.dumps,
        )
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.console = Console()

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def post_request(self, document):
        response = self.session.post(
            self.endpoint + document["id"], data=ujson.dumps(document["fields"])
        )
        response.raise_for_status()
        return len(ujson.dumps(document["fields"]).encode("utf-8"))

    async def post_request_async(self, document):
        async with self.async_session.post(
            self.endpoint + document["id"], json=document["fields"]
        ) as response:
            response.raise_for_status()
        return len(ujson.dumps(document["fields"]).encode("utf-8"))

    def ping_test(self):
        try:
            response = self.session.get(self.endpoint)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def feed_sync(self):
        with Progress(
            TextColumn("[bold blue]{task.fields[mode]}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TransferSpeedColumn(),
            "•",
            "[bold green]{task.completed}/{task.total}",
            console=self.console,
        ) as progress:
            task = progress.add_task(
                "Feeding documents synchronously...",
                total=len(self.documents),
                mode="Sync",
                start=False,
            )
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self.post_request, document)
                    for document in self.documents
                ]
                progress.start_task(task)
                for future in futures:
                    progress.update(task, advance=future.result())

    async def feed_async(self):
        with Progress(
            TextColumn("[bold blue]{task.fields[mode]}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TransferSpeedColumn(),
            "•",
            "[bold green]{task.completed}/{task.total}",
            console=self.console,
        ) as progress:
            task = progress.add_task(
                "Feeding documents asynchronously...",
                total=len(self.documents),
                mode="Async",
                start=False,
            )
            semaphore = asyncio.Semaphore(self.max_connections)

            async def post_with_semaphore(document):
                async with semaphore:
                    return await self.post_request_async(document)

            tasks = [
                asyncio.ensure_future(post_with_semaphore(document))
                for document in self.documents
            ]
            progress.start_task(task)
            for future in asyncio.as_completed(tasks):
                result = await future
                progress.update(task, advance=result)

    async def feed_async_wrapper(self):
        await self.feed_async()

    def feed(self):
        if self.ping_test():
            asyncio.run(self.feed_async_wrapper())
        else:
            self.feed_sync()

    def __del__(self):
        self.session.close()
        self.async_session.close()
        self.executor.shutdown()
