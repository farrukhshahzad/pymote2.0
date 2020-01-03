.. image:: https://api.travis-ci.org/darbula/pymote.png?branch=master
  :target: http://travis-ci.org/darbula/pymote

.. image:: https://coveralls.io/repos/darbula/pymote/badge.png?branch=master
  :target: https://coveralls.io/r/darbula/pymote?branch=master


Pymote
======

Pymote is a Python package for event based simulation and evaluation of distributed algorithms.

Definition of distributed environment used as specifications for making Pymote are taken mainly from `Design and Analysis of Distributed Algorithms <http://eu.wiley.com/WileyCDA/WileyTitle/productCd-0471719978,descCd-description.html>`_ by Nicola Santoro.

Pymote's main goal is to provide framework for fast implementation and simulation of distributed algorithms. In its current initial state it is mainly targeted at algorithms for wireless (sensor) networks but it could be applied to any distributed computing environment (e.g., distributed systems, grid networks, internet, etc.).

.. figure:: docs/install/_images/pymote_console_gui.png
   :align: center
   
   Pymote is being developed on top of `NetworkX <https://github.com/networkx/networkx/>`_ and is ment to be used along other scientific packages such as SciPy, NumPy and matplotlib. Currently, gui runs on PySide (Qt bindings) and console is jazzy IPython.

Installation
------------

For installation instructions please visit `documentation <https://pymote.readthedocs.org>`_.

Literature
----------

Santoro, N.: *Design and Analysis of Distributed Algorithms*, 2006 `link <http://eu.wiley.com/WileyCDA/WileyTitle/productCd-0471719978,descCd-description.html>`_

Arbula, D. and Lenac, K.: *Pymote: High Level Python Library for Event-Based Simulation and Evaluation of Distributed Algorithms*, International Journal of Distributed Sensor Networks, Volume 2013 `link <http://www.hindawi.com/journals/ijdsn/2013/797354/>`_

Pymote 2.0
=========
PhD dissertation work by Farrukh Shahzad.

F. Shahzad, TR Sheltami, EM Shakshuki:  *DV-maxHop: A fast and accurate range-free localization algorithm for anisotropic wireless networks*, IEEE Transactions on Mobile Computing, 2016 `https://ieeexplore.ieee.org/abstract/document/7756390/`

A.	Propagation Model
---------------------
We implemented two basic radio propagation models and the commonly used shadowing model for WSN in the Pymote framework. These models are used to predict the received signal power of each packet. At the physical layer of each wireless node, there is a receiving threshold (P_RX_THRESHOLD). When a packet is received, if its signal power is below the receiving threshold, it is marked as error and dropped by the MAC layer. The free space propagation model assumes the ideal propagation condition that there is only one clear line-of-sight path between the transmitter and receiver, while the two-ray ground reflection model considers both the direct path and a ground reflection path which gives more accurate prediction at a long distance than the free space model. However, in reality, the received power is a random variable due to multipath propagation or fading (shadowing) effects. The shadowing model consists of two parts: path loss component and a Gaussian random variable with zero mean and standard deviation ?DB, which represent the variation of the received power at certain distance. Table I lists parameters available for propagation module. The propagation model type (free space, two-ray ground or shadowing) is a network level attribute, which should be selected before starting the simulation.

B.	Energy consumption Model
---------------------
In our extended framework, the energy model object is implemented as a node attribute, which represents the level of energy in a node. Each node can be configured to be powered by external source (unlimited power), Battery (default) or energy harvesting (EH) sources. The energy in a node has an initial value which is the level of energy the node has at the beginning of the simulation. It also has a given energy consumption for every packet it transmits and receives which is a function of packet size, transmission rate and transmit (receive) power. The model also supports idle or constant energy discharge due to hardware/ microcontroller consumption and energy charge for energy harvesting based WSN. During simulation, each node’s available energy is recomputed every second based on the charging and/or discharging rate. If it drops below minimum energy required to operate (Emin) then that node assumed to be dead (not available for communication) until energy reaches above Emin again later in simulation (for EH nodes). Table II lists parameters available for energy module which can be set differently for each node. The energy object keeps track of the energy available (for battery-operated or energy harvested nodes) and total energy consumption.

C.	Mobility Model
---------------------

Our extended framework allows nodes to be mobile during simulation. Each node can be configured as fixed or mobile. The mobility module support three types of motion as summarized in Table III. During simulation, each mobile node location is recomputed every second.

D.	Plotting and Data collection
---------------------
These modules allow real-time plotting and data collection during and after simulation for interactive analysis and comparisons of useful information. The modules implements generic helper methods. The simulation script is responsible for utilizing these methods to plot/chart and collect/log appropriate information as required by the simulated algorithm and application scenario. The output files are managed by utilizing separate folder for each type of files within the current working path (Table IV). Also for each simulation run, a separate folder, prefixed with the current date time is used for all files created during that simulation run.

E.	Modified Node module
---------------------
Enhanced framework requires significant modification in the Node module. The Node object now contains node type, energy model object and mobility object. The modified send and receive methods check before transmission or reception whether node has enough energy to perform the operation. Also the propagation model dictates whether a packet is received without errors (i.e. when received signal power is greater than the threshold based on the distance between the sender and receiver nodes). The object also keeps track of number of messages transmitted, received, or lost.
