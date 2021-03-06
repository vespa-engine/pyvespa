Reference API
=============

Create an Application Package
-----------------------------

The first step to create a Vespa application is to create an instance of :class:`ApplicationPackage`.

ApplicationPackage
++++++++++++++++++

.. autoclass:: vespa.package.ApplicationPackage
   :members:
   :special-members: __init__

Schema and Document
-------------------

An :class:`ApplicationPackage` instance comes with a default :class:`Schema` that contains a default :class:`Document`,
meaning that you usually do not need to create those yourself.

Schema
++++++

.. autoclass:: vespa.package.Schema
   :members:
   :special-members: __init__

Document
++++++++

.. autoclass:: vespa.package.Document
   :members:
   :special-members: __init__

Field
+++++

Once we have an :class:`ApplicationPackage` instance containing a :class:`Schema` and a :class:`Document` we usually
want to add fields so that we can store our data in a structured manner. We can accomplish that by creating
:class:`Field` instances and adding those to the :class:`ApplicationPackage` instance via :class:`Schema` and
:class:`Document` methods.

.. autoclass:: vespa.package.Field
   :members:
   :special-members: __init__

FieldSet
++++++++

.. autoclass:: vespa.package.FieldSet
   :members:
   :special-members: __init__

RankProfile
+++++++++++

.. autoclass:: vespa.package.RankProfile
   :members:
   :special-members: __init__

Query Profile
-------------

A :class:`QueryProfile` is a named collection of search request parameters given in the configuration. The search
request can specify a query profile whose parameters will be used as parameters of that request. The query profiles may
optionally be type checked. Type checking is turned on by referencing a :class:`QueryProfileType` from the query
profile.

An :class:`ApplicationPackage` instance comes with a default :class:`QueryProfile` named `default` that is associated
with a :class:`QueryProfileType` named `root`, meaning that you usually do not need to create those yourself, only add
fields to them when required.

Create a QueryProfileType
+++++++++++++++++++++++++

QueryTypeField
~~~~~~~~~~~~~~

.. autoclass:: vespa.package.QueryTypeField
   :members:
   :special-members: __init__

QueryProfileType
~~~~~~~~~~~~~~~~

.. autoclass:: vespa.package.QueryProfileType
   :members:
   :special-members: __init__

Create a QueryProfile
+++++++++++++++++++++

QueryField
~~~~~~~~~~

.. autoclass:: vespa.package.QueryField
   :members:
   :special-members: __init__

QueryProfile
~~~~~~~~~~~~

.. autoclass:: vespa.package.QueryProfile
   :members:
   :special-members: __init__

Deploying your application
++++++++++++++++++++++++++

VespaDocker
~~~~~~~~~~~

.. autoclass:: vespa.package.VespaDocker
   :members:
   :special-members: __init__

VespaCloud
~~~~~~~~~~

.. autoclass:: vespa.package.VespaCloud
   :members:
   :special-members: __init__

Query Model
-----------

A :class:`~vespa.query.QueryModel` is an abstraction that encapsulates all the relevant information controlling
how your app match and rank documents. A `QueryModel` can be used for querying (:func:`~vespa.application.Vespa.query`),
evaluating (:func:`~vespa.application.Vespa.evaluate`) and collecting data
(:func:`~vespa.application.Vespa.collect_training_data`) from your app.

Create a QueryModel
+++++++++++++++++++

.. autoclass:: vespa.query.QueryModel
   :members:
   :special-members: __init__

Match phase
+++++++++++

Union
~~~~~

.. autoclass:: vespa.query.Union
   :members:
   :special-members: __init__

AND
~~~

.. autoclass:: vespa.query.AND
   :members:
   :special-members: __init__

OR
~~

.. autoclass:: vespa.query.OR
   :members:
   :special-members: __init__

WeakAnd
~~~~~~~

.. autoclass:: vespa.query.WeakAnd
   :members:
   :special-members: __init__

ANN
~~~

.. autoclass:: vespa.query.ANN
   :members:
   :special-members: __init__

Rank Profile
++++++++++++

RankProfile
~~~~~~~~~~~

.. autoclass:: vespa.query.RankProfile
   :members:
   :special-members: __init__

Query Properties
++++++++++++++++

QueryRankingFeature
~~~~~~~~~~~~~~~~~~~

.. autoclass:: vespa.query.QueryRankingFeature
   :members:
   :special-members: __init__

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

