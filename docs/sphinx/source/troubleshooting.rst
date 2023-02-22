Troubleshooting
===============

Also see the `Vespa FAQ <https://docs.vespa.ai/en/faq.html>`__
and `Vespa support <https://vespa.ai/support>`__ for more help resources.


Vespa.ai and pyvespa
--------------------
Both `Vespa <https://vespa.ai/>`__ and pyvespa APIs change regularly -
make sure to use the latest version of `vespaengine/vespa <https://hub.docker.com/r/vespaengine/vespa>`__
by running ``docker pull vespaengine/vespa`` and :doc:`install pyvespa <index>`.



Port conflicts / Docker
-----------------------
Some of the notebooks run a Docker container.
Make sure to stop running Docker containers before (re)running pyvespa notebooks -
run ``docker ps`` and ``docker ps -a -q -f status=exited`` to list containers.

pyvespa will start a Docker container with 4G memory by default -
make sure Docker settings have at least this.



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
Also see `Feed block <https://docs.vespa.ai/en/operations/feed-block.html>`__ -
Vespa stops writes before the disk goes full.



Too many open files during batch feeding
----------------------------------------
This is an OS-related issue. There are two options to solve the problem:

1. Reduce the number of async connections via the connections parameter:
   ``app.feed_batch(..., connections, ...)``.

2. Increase the open file limit: ``ulimit -n 10000``.
   Check if the limit was increased with ``ulimit -Sn``.
