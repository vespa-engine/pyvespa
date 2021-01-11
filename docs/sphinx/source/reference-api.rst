Reference API
=============

Create an Application Package
-----------------------------

The first step to create a Vespa application is to create an instance of :class:`ApplicationPackage`.

.. autoclass:: vespa.package.ApplicationPackage
   :members:
   :special-members: __init__

Schema and Document
-------------------

An :class:`ApplicationPackage` instance comes with a default :class:`Schema` that contains a default :class:`Document`,
meaning that you usually do not need to create those yourself.

.. autoclass:: vespa.package.Schema
   :members:
   :special-members: __init__

.. autoclass:: vespa.package.Document
   :members:
   :special-members: __init__

Create a Field
++++++++++++++

Once we have an :class:`ApplicationPackage` instance containing a :class:`Schema` and a :class:`Document` we usually
want to add fields so that we can store our data in a structured manner. We can accomplish that by creating
:class:`Field` instances and adding those to the :class:`ApplicationPackage` instance via :class:`Schema` and
:class:`Document` methods.

.. autoclass:: vespa.package.Field
   :members:
   :special-members: __init__

Create a FieldSet
+++++++++++++++++

.. autoclass:: vespa.package.FieldSet
   :members:
   :special-members: __init__

Create a RankProfile
++++++++++++++++++++

.. autoclass:: vespa.package.RankProfile
   :members:
   :special-members: __init__

Query Profile
-------------



vespa.application module
------------------------

.. automodule:: vespa.application
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

vespa.evaluation module
-----------------------

.. automodule:: vespa.evaluation
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

vespa.package module
--------------------

.. automodule:: vespa.package
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

vespa.query module
------------------

.. automodule:: vespa.query
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
