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

The best way to get started is by following the tutorials below. You can easily run them yourself on Google Colab
by clicking on the badge at the top of the tutorial.


- :doc:`connect-to-vespa-instance`
- :doc:`create-and-deploy-vespa-cloud`
- :doc:`create-and-deploy-vespa-docker`


How-to guides
+++++++++++++

- :doc:`application-package`
- :doc:`deploy-application`
- :doc:`query-model`
- :doc:`query`
- :doc:`evaluation`
- :doc:`collect-training-data`

Use cases
+++++++++

- :doc:`use_cases/cord19/cord19`
- :doc:`use_cases/msmarco/msmarco`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
