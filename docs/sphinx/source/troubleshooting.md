# Troubleshooting

Also see the [Vespa FAQ](https://docs.vespa.ai/en/faq.html) and [Vespa support](https://cloud.vespa.ai/support) for more help resources.

## Vespa.ai and pyvespa

Both [Vespa](https://vespa.ai/) and pyvespa APIs change regularly - make sure to use the latest version of [vespaengine/vespa](https://hub.docker.com/r/vespaengine/vespa) by running `docker pull vespaengine/vespa` and [install pyvespa](https://vespa-engine.github.io/pyvespa).

To check the current version, run:

```bash
python3 -m pip show pyvespa
```

## Docker Memory

pyvespa will start a Docker container with 4G memory by default - make sure Docker settings have at least this. Use the Docker Desktop settings or `docker info | grep "Total Memory"` or `podman info | grep "memTotal"` to validate.

## Port conflicts / Docker

Some of the notebooks run a Docker container. Make sure to stop running Docker containers before (re)running pyvespa notebooks - run `docker ps` and `docker ps -a -q -f status=exited` to list containers.

## Deployment

Vespa has safeguards for incompatible deployments, and will warn with *validation-override* or *INVALID_APPLICATION_PACKAGE* in the deploy output. See [validation-overrides](https://docs.vespa.ai/en/reference/validation-overrides.html). This is most often due to pyvespa reusing a Docker container instance. The fix is to list (`docker ps`) and remove (`docker rm -f <container id>`) the existing Docker containers. Alternatively, use the Docker Dashboard application. Then deploy again.

After deployment, validate status:

* Config server state: [http://localhost:19071/state/v1/health](http://localhost:19071/state/v1/health)
* Container state: [http://localhost:8080/state/v1/health](http://localhost:8080/state/v1/health)

Look for `"status" : { "code" : "up"}` - both URLs must work before feeding or querying.

## Full disk

Make sure to allocate enough disk space for Docker in Docker settings. If writes/queries fail or return no results, look in the `vespa.log` (output in the Docker dashboard):

```
WARNING searchnode
proton.proton.server.disk_mem_usage_filter   Write operations are now blocked:
'diskLimitReached: { action: "add more content nodes",
reason: "disk used (0.939172) > disk limit (0.9)",
stats: { capacity: 50406772736, used: 47340617728, diskUsed: 0.939172, diskLimit: 0.9}}'
```

Future pyvespa versions might throw an exception in these cases. See [Feed block](https://docs.vespa.ai/en/operations/feed-block.html) - Vespa stops writes before the disk goes full. Add more disk space, clean up, or follow the [example](https://vespa-engine.github.io/pyvespa/application-packages.html#Deploy-from-modified-files) to reconfigure for higher usage.

## Check number of indexed documents

For query errors, check the number of documents indexed before debugging further:

```python
app.query(yql='select * from sources * where true').number_documents_indexed
```

If this is zero, check that the deployment of the application worked, and that the subsequent feeding step completed successfully.

## Too many open files during batch feeding

This is an OS-related issue. There are two options to solve the problem:

1. Reduce the number of connections via the `connections` parameter:
   ```python
   with app.syncio(connections=12):
   ```

2. Increase the open file limit: `ulimit -n 10000`. Check if the limit was increased with `ulimit -Sn`.

## Data export

`vespa visit` exports data from Vespa - see [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html#documents). Use this to validate data feeding and troubleshoot query issues.
