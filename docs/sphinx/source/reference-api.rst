Reference API
=============

Define stateful application
***************************

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

Define stateless application
****************************

Create tasks
------------

SequenceClassification
++++++++++++++++++++++

.. autoclass:: vespa.ml.SequenceClassification
   :members:
   :special-members: __init__

Create model server
-------------------

ModelServer
+++++++++++

.. autoclass:: vespa.package.ModelServer
   :members:
   :special-members: __init__

Deploy your application
***********************

Deploy your stateful or stateless applications.

VespaDocker
-----------

.. autoclass:: vespa.deployment.VespaDocker
   :members:
   :special-members: __init__

VespaCloud
----------

.. autoclass:: vespa.deployment.VespaCloud
   :members:
   :special-members: __init__

Connect to existing application
*******************************

Vespa
-----

.. autoclass:: vespa.application.Vespa
   :special-members: __init__

Interact to existing application
********************************

Feed data
---------

feed_batch
++++++++++

.. automethod:: vespa.application.Vespa.feed_batch

feed_data_point
+++++++++++++++

.. automethod:: vespa.application.Vespa.feed_data_point

Get, update and delete data
---------------------------

get_data
++++++++

.. automethod:: vespa.application.Vespa.get_data

get_batch
+++++++++

.. automethod:: vespa.application.Vespa.get_batch

update_data
+++++++++++

.. automethod:: vespa.application.Vespa.update_data

update_batch
++++++++++++

.. automethod:: vespa.application.Vespa.update_batch

delete_data
+++++++++++

.. automethod:: vespa.application.Vespa.delete_data

delete_batch
++++++++++++

.. automethod:: vespa.application.Vespa.delete_batch

delete_all_docs
+++++++++++++++

.. automethod:: vespa.application.Vespa.delete_all_docs

Query
-----

.. automethod:: vespa.application.Vespa.query

Run experiments
---------------

evaluate
++++++++

.. automethod:: vespa.application.Vespa.evaluate

evaluate_query
++++++++++++++

.. automethod:: vespa.application.Vespa.evaluate_query

Collect training data
---------------------

collect_training_data
+++++++++++++++++++++

.. automethod:: vespa.application.Vespa.collect_training_data

collect_training_data_point
+++++++++++++++++++++++++++

.. automethod:: vespa.application.Vespa.collect_training_data_point

Query Model
***********

A :class:`~vespa.query.QueryModel` is an abstraction that encapsulates all the relevant information controlling
how your app match and rank documents. A `QueryModel` can be used for querying (:func:`~vespa.application.Vespa.query`),
evaluating (:func:`~vespa.application.Vespa.evaluate`) and collecting data
(:func:`~vespa.application.Vespa.collect_training_data`) from your app.

Create a QueryModel
-------------------

.. autoclass:: vespa.query.QueryModel
   :members:
   :special-members: __init__

Match phase
-----------

Union
+++++

.. autoclass:: vespa.query.Union
   :members:
   :special-members: __init__

AND
+++

.. autoclass:: vespa.query.AND
   :members:
   :special-members: __init__

OR
++

.. autoclass:: vespa.query.OR
   :members:
   :special-members: __init__

WeakAnd
+++++++

.. autoclass:: vespa.query.WeakAnd
   :members:
   :special-members: __init__

ANN
+++

.. autoclass:: vespa.query.ANN
   :members:
   :special-members: __init__

Rank Profile
------------

RankProfile
+++++++++++

.. autoclass:: vespa.query.RankProfile
   :members:
   :special-members: __init__

Query Properties
----------------

QueryRankingFeature
+++++++++++++++++++

.. autoclass:: vespa.query.QueryRankingFeature
   :members:
   :special-members: __init__

Evaluation Metrics
******************

MatchRatio
----------

.. autoclass:: vespa.evaluation.MatchRatio
   :members:
   :special-members: __init__

Recall
------

.. autoclass:: vespa.evaluation.Recall
   :members:
   :special-members: __init__

ReciprocalRank
--------------

.. autoclass:: vespa.evaluation.ReciprocalRank
   :members:
   :special-members: __init__

NormalizedDiscountedCumulativeGain
----------------------------------

.. autoclass:: vespa.evaluation.NormalizedDiscountedCumulativeGain
   :members:
   :special-members: __init__
