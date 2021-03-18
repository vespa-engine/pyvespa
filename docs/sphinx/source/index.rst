.. pyvespa documentation master file, created by
   sphinx-quickstart on Wed Aug 26 11:11:55 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Vespa python API
================

.. toctree::
   :hidden:

   install
   quickstart
   howto
   usecases
   reference-api

Vespa_ is the scalable open-sourced serving engine that enable us to store, compute and rank big data at user
serving time. ``pyvespa`` provides a python API to Vespa. It allow us to create, modify, deploy and interact with
running Vespa instances. The main goal of the library is to allow for faster prototyping and to facilitate
Machine Learning experiments for Vespa applications.

.. _Vespa: https://vespa.ai/


Install
+++++++

.. warning::
    The library is under active development and backward incompatible changes may occur.

You can install ``pyvespa`` via ``pip``:

.. code:: bash

	pip install pyvespa

Quick-start
+++++++++++

There are three ways you can get value out of pyvespa:

#. You can connect to a running Vespa application.
#. You can build and deploy a Vespa application using pyvespa API.
#. You can deploy an application from Vespa config files stored on disk.

Read more:

- :doc:`three-ways-to-get-started-with-pyvespa`

Follow the tutorials below for a more in-depth treatment of each of the methods mentioned above.

- :doc:`connect-to-vespa-instance`
- :doc:`create-and-deploy-vespa-cloud`
- :doc:`create-and-deploy-vespa-docker`

How-to guides
+++++++++++++

- :doc:`howto/create_app_package/create_app_package`
- :doc:`howto/deploy_app_package/deploy_app_package`
- :doc:`query-model`
- :doc:`query`
- :doc:`evaluation`
- :doc:`collect-training-data`

Use cases
+++++++++

- :doc:`use_cases/cord19/cord19`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
