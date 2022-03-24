.. pyvespa documentation master file, created by
   sphinx-quickstart on Wed Aug 26 11:11:55 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Vespa python API
================

.. toctree::
   :hidden:

   install
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

Use pyvespa to:

#. Build and deploy a Vespa application using pyvespa API.
#. Connect to an existing Vespa application and run queries from python.
#. Import a Vespa application package from files and use pyvespa to access it.

Read more:

- :doc:`overview`
