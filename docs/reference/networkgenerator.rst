.. _networkgenerator:

Network Generator
=================
Implementation of different methods for automated network creation.
It defines parameters (conditions) that generated network must satisfy.

.. currentmodule:: pymote.networkgenerator
.. automodule:: pymote.networkgenerator
.. autoclass:: NetworkGenerator

Methods
-------
.. automethod:: pymote.networkgenerator.NetworkGenerator.generate_random_network
.. automethod:: pymote.networkgenerator.NetworkGenerator.generate_neigborhood_network

Default procedure
-----------------
For any generator method network attributes take default priorities
which are defined like this:
 
* first network is created in given environment with `n_count` number
  of nodes and `comm_range` communication range
* if `connected` is True it must be satisfied, if not satisfied initially:
    * gradually increase number of nodes up to `n_max`
    * if comm_range is None gradually increase nodes commRange
    * if still not connected raise an exception
* if `degree` condition is defined and current network degree is
    * lower - repeat measures from the last step to increase current
      network degree
    * higher one degree or more - try countermeasures i.e. decrease number of
      nodes and commRange but without influencing other defined and already
      satisfied parameters (`connected`)
