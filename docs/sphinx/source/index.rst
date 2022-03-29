.. pyvespa documentation master file, created by
   sphinx-quickstart on Wed Aug 26 11:11:55 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Vespa python API
================

.. toctree::
   :hidden:

   overview
   quickstart
   usecases
   reference-api


.. image:: https://vespa.ai/assets/vespa-logo-color.png

Vespa_ is the scalable open-sourced serving engine to store, compute and rank big data at user serving time.
``pyvespa`` provides a python API to Vespa -
use it to create, modify, deploy and interact with running Vespa instances.
The main goal of the library is to allow for faster prototyping
and to facilitate Machine Learning experiments for Vespa applications.

.. _Vespa: https://vespa.ai/


.. warning::
    The library is under active development and backward incompatible changes may occur.

Install ``pyvespa`` via ``pip``:

.. code:: bash

	pip install pyvespa

pyvespa provides a python API to Vespa.
The library’s primary goal is to allow for faster prototyping
and facilitate Machine Learning experiments for Vespa applications:

#. Build and deploy a Vespa application using pyvespa API.
#. Connect to an existing Vespa application and run queries from python.
#. Import a Vespa application package from files and use pyvespa to access it.


Build and deploy
****************
The `getting-started-pyvespa <getting-started-pyvespa.html>`_ notebook
is a good primer on how to create an application, feed data and run queries.


Query a running Vespa application
*********************************
When a Vespa application is already running,
one can instantiate the `Vespa <reference-api.html#vespa.application.Vespa>`_ class with the endpoint.
Refer to `connect-to-vespa-instance <connect-to-vespa-instance.html>`_ to connect to and application and run queries.

Deploy from Vespa config files
******************************
Use pyvespa to `deploy a Vespa application package <overview.html>`_ to a local Docker container.
