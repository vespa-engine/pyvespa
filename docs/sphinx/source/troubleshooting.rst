Troubleshooting
===============

Also see the `Vespa FAQ <https://docs.vespa.ai/en/faq.html>`__
and `Vespa support <https://vespa.ai/support>`__ for more help resources.


Vespa.ai and pyvespa
--------------------
Both `Vespa <https://vespa.ai/>`__ and pyvespa APIs change regularly -
make sure to use the latest version of `vespaengine/vespa <https://hub.docker.com/r/vespaengine/vespa>`__
by running ``docker pull vespaengine/vespa``
and `pyvespa <getting-started-pyvespa.html>`__.



Port conflicts / Docker
-----------------------
Some of the notebooks run a Docker container.
Make sure to stop running Docker containers before (re)running pyvespa notebooks -
run ``docker ps`` and ``docker ps -a -q -f status=exited`` to list containers.

pyvespa will start a Docker container with 4G memory by default -
make sure Docker settings have at least this.



Too many open files during batch feeding
----------------------------------------
This is an OS-related issue. There are two options to solve the problem:

1. Reduce the number of async connections via the connections parameter:
   ``app.feed_batch(..., connections, ...)``.

2. Increase the open file limit: ``ulimit -n 10000``.
   Check if the limit was increased with ``ulimit -Sn``.
