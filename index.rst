.. pyvespa documentation master file, created by
   sphinx-quickstart on Wed Aug 26 11:11:55 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. image:: https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg


Vespa python API
================

.. toctree::
   :hidden:

   getting-started-pyvespa
   getting-started-pyvespa-cloud
   advanced-configuration
   authenticating-to-vespa-cloud
   application-packages
   query
   reads-writes
   evaluating-vespa-application-cloud
   reference-api
   troubleshooting
   examples

`Vespa <https://vespa.ai/>`__ is the scalable open-sourced serving engine to store,
compute and rank big data at user serving time.
``pyvespa`` provides a python API to Vespa -
use it to create, modify, deploy and interact with running Vespa instances.
The main goal of the library is to allow for faster prototyping and get familiar with Vespa features.

.. warning::
    pyvespa is under active development and backward incompatible changes may occur.

:doc:`getting-started-pyvespa` is a good primer on how to create an application, feed data and run queries.
See :doc:`examples` for use cases.
The following blog post series will get you started:

* `Run search engine experiments in Vespa from python <https://blog.vespa.ai/run-search-engine-experiments-in-Vespa-from-python/>`__

* `Build sentence/paragraph level QA application from python with Vespa <https://blog.vespa.ai/build-qa-app-from-python-with-vespa/>`__

* `Build a basic text search application from python with Vespa: Part 1 <https://blog.vespa.ai/build-basic-text-search-app-from-python-with-vespa/>`__

* `Build a News recommendation app from python with Vespa: Part 1 <https://blog.vespa.ai/build-news-search-app-from-python-with-vespa/>`__

The `Vespa FAQ <https://docs.vespa.ai/en/faq.html>`__ is a great resource,
also see :doc:`pyvespa troubleshooting <troubleshooting>`.


Requirements
************
Install ``pyvespa``:

.. code:: bash

	$ python3 -m pip install pyvespa

Install `jupyter notebook <https://jupyter.org/install#jupyter-notebook>`__
to run the notebooks in a browser:

.. code:: bash

    $ git clone --depth 1 https://github.com/vespa-engine/pyvespa.git
    $ jupyter notebook --notebook-dir pyvespa/docs/sphinx/source

Many of the pyvespa guides / notebooks use Docker -
minimum memory requirement is 4 Gb unless other documented:

.. code:: bash

    $ docker info | grep "Total Memory"
    or
    $ podman info | grep "memTotal"

One can also use :doc:`Vespa Cloud <getting-started-pyvespa-cloud>` to run the notebooks.
