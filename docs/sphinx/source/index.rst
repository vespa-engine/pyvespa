.. pyvespa documentation master file, created by
   sphinx-quickstart on Wed Aug 26 11:11:55 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Vespa python API
================

.. toctree::
   :hidden:

   getting-started-pyvespa
   query
   query-model
   deploy-docker
   deploy-vespa-cloud
   exchange-data-with-app
   collect-training-data
   evaluation
   learning-to-rank-ignore
   usecases
   reference-api


.. image:: https://vespa.ai/assets/vespa-logo-color.png

`Vespa <https://vespa.ai/>`__ is the scalable open-sourced serving engine to store,
compute and rank big data at user serving time.
``pyvespa`` provides a python API to Vespa -
use it to create, modify, deploy and interact with running Vespa instances.
The main goal of the library is to allow for faster prototyping
and to facilitate Machine Learning experiments for Vespa applications.

.. warning::
    The library is under active development and backward incompatible changes may occur.

pyvespa provides a python API to Vespa.
The library’s primary goal is to allow for faster prototyping
and facilitate Machine Learning experiments for Vespa applications:

#. Build and deploy a Vespa application using pyvespa API.
#. Connect to an existing Vespa application and run queries from python.
#. Import a Vespa application package from files and use pyvespa to access it.


Build and deploy
****************
The `getting-started-pyvespa <getting-started-pyvespa.rst>`__ notebook
is a good primer on how to create an application, feed data and run queries.


Query a running Vespa application
*********************************
When a Vespa application is already running,
one can instantiate the `Vespa <reference-api.rst#vespa.application.Vespa>`__ class with the endpoint.
Refer to `query application <query.rst>`__ to connect to an application and run queries.


Deploy from Vespa config files
******************************
Use pyvespa to `deploy a Vespa application package <deploy-docker.rst>`__
to a local Docker container.


Requirements
************
This documentation assumes ``pyvespa`` is installed:

.. code:: bash

	$ python3 -m pip install pyvespa

Install `jupyter notebook <https://jupyter.org/install#jupyter-notebook>`__
to run the notebooks in the browser:

.. code:: bash

    $ git clone --depth 1 https://github.com/vespa-engine/pyvespa.git
    $ jupyter notebook --notebook-dir pyvespa/docs/sphinx/source

Many of the pyvespa guides / notebooks use Docker -
minimum memory requirement is 4 Gb unless other documented:

.. code:: bash

    $ docker info | grep "Total Memory"

One can also use `Vespa Cloud <deploy-vespa-cloud.rst>`__ to run the notebooks.

The `Vespa FAQ <https://docs.vespa.ai/en/faq.html>`__ is a great resource for troubleshooting.
