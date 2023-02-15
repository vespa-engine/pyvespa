Troubleshooting
===============


Too many open files during batch feeding
----------------------------------------

This is an OS related issue. There are two options to solve:

1. Reduce the number of async connections via the connections parameter:
   ``app.feed_batch(..., connections, ...)``

2. Increase the open file limit: ``ulimit -n 10000``.
   Check if the limit was increased with ``ulimit -Sn``.
