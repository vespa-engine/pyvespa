Troubleshooting
===============

Also see the `Vespa FAQ <https://docs.vespa.ai/en/faq.html>`__
and `Vespa support <https://cloud.vespa.ai/support>`__ for more help resources.


Vespa.ai and pyvespa
--------------------
Both `Vespa <https://vespa.ai/>`__ and pyvespa APIs change regularly -
make sure to use the latest version of `vespaengine/vespa <https://hub.docker.com/r/vespaengine/vespa>`__
by running ``docker pull vespaengine/vespa`` and :doc:`install pyvespa <index>`.

``python3 -m pip show pyvespa`` shows current version.


Docker Memory
-------------
pyvespa will start a Docker container with 4G memory by default -
make sure Docker settings have at least this.
Use the Docker Desktop settings or ``docker info | grep "Total Memory"`` or ``podman info | grep "memTotal"`` to validate.


Port conflicts / Docker
-----------------------
Some of the notebooks run a Docker container.
Make sure to stop running Docker containers before (re)running pyvespa notebooks -
run ``docker ps`` and ``docker ps -a -q -f status=exited`` to list containers.



Deployment
----------
Vespa has safeguards for incompatible deployments,
and will warn with *validation-override* or *INVALID_APPLICATION_PACKAGE* in the deploy output.
See `validation-overrides <https://docs.vespa.ai/en/reference/validation-overrides.html>`__.
Most often is this due to pyvespa reusing a Docker container instance,
and the fix is to list - ``docker ps`` - and remove  - ``docker rm -f <container id>`` -
the existing Docker containers.
Alternatively, using the Docker Dashboard application.
Then deploy again.

After a deployment, validate status:

* Config server state: http://localhost:19071/state/v1/health
* Container state: http://localhost:8080/state/v1/health

Look for ``"status" : { "code" : "up"}`` - both URLs must work before feeding or querying.

Full disk
---------
Make sure to allocate enough disk space for Docker in Docker settings -
if writes/queries fail/no results, look in the vespa.log (output in the Docker dashboard):

``WARNING searchnode
proton.proton.server.disk_mem_usage_filter   Write operations are now blocked:
'diskLimitReached: { action: "add more content nodes",
reason: "disk used (0.939172) > disk limit (0.9)",
stats: { capacity: 50406772736, used: 47340617728, diskUsed: 0.939172, diskLimit: 0.9}}'``

Future pyvespa versions might throw an exception in these cases.
See `Feed block <https://docs.vespa.ai/en/operations/feed-block.html>`__ -
Vespa stops writes before the disk goes full.
Add more disk / clean up, or follow the
`example <https://vespa-engine.github.io/pyvespa/application-packages.html#Deploy-from-modified-files>`__
to reconfigure for higher usage.



Check number of indexed documents
---------------------------------
For query errors, check the number of documents indexed before debugging further:
``app.query(yql='select * from sources * where true).number_documents_indexed``.

If this is zero, check that the deployment of the application worked, and the subsequent feeding step.

Too many open files during batch feeding
----------------------------------------
This is an OS-related issue. There are two options to solve the problem:

1. Reduce the number of connections via the connections parameter:
   ``with app.syncio(connections=12):``.

2. Increase the open file limit: ``ulimit -n 10000``.
   Check if the limit was increased with ``ulimit -Sn``.

Data export
-----------
``vespa visit`` exports data from Vespa - see `Vespa CLI <https://docs.vespa.ai/en/vespa-cli.html#documents>`__.
Use this to validate data feeding and troubleshoot query issues.
