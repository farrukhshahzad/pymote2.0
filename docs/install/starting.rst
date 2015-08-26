Starting Pymote
===============
Pymote features interactive console based on IPython and simulation GUI.


.. figure:: _images/pymote_console_gui.png
   :align: center
   
   Pymote console and GUI
   
Interactive console (IPython)
-----------------------------
To use Pymote from the interactive console (IPython) start provided program 
``ipymote`` with previously activated virtual environment::
    
    > ipymote

.. note::

    If virtualenv is used make sure that virtual environment is activated (:ref:`linux <linux-venvact>`, :ref:`windows <windows-venvact>`) and if WinPython is used then run ``ipymote`` from WinPython Command Prompt.

Pymote can also be started by starting IPython directly and using dedicated ``pymote`` profile::

    > ipython --profile=pymote

.. note::

    Pymote profile files should be present inside 
    ``pymote_env/.ipython/profile_pymote/ipython_config.py``
    or ``~/.ipython/profile_pymote/ipython_config.py`` file created during Pymote installation.


Simulation GUI
--------------
Pymote features simulation GUI which can be started as standalone application using 
``pymote-simgui`` (in Windows ``pymote-simgui.exe``). 

.. note::

    If pymote is installed in virtual environment then `pymote-simgui` starts inside this 
    environment. When network pickle is opened in simulator all algorithms this network is 
    referencing must be importable from virtual environment. The easy and proper way to ensure that
    the algorithms are importable is to use bootstrap algorithms package that can be found in 
    `pymote-algorithms-bootstrap <https://github.com/darbula/pymote-algorithms-bootstrap>`_ 
    and follow the instructions found there.
    

Simulation GUI running from the interactive console
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Very convenient way of starting and working with the GUI is from the interactive console by running ``simulationgui.py`` like this::

    In [1]: %run path/to/pymote/gui/simulationgui.py

The gui event loop is separated from the console. Simulation window can be accessed by using ``simgui`` and network in the simulator window by using ``simgui.net`` so all simulation objects (network, nodes, messages...) are fully inspectable and usable via console.



