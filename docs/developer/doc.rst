Writing documentation
*********************
This section describes certain not obvious details in writing documentation for Pymote in sphinx.


Intersphinx
===========

To auto-reference external document in with intersphinx:

1. set `intersphinx_mapping <http://sphinx-doc.org/ext/intersphinx.html#confval-intersphinx_mapping>`_ in ``conf.py``::

        intersphinx_mapping = {
            'python': ('http://docs.python.org/', None),
            'numpy': ('http://docs.scipy.org/doc/numpy/', None),
            'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
        }

2. reference in docs with ```:py:<type>:`<ref>``` i.e. ``:py:class:`numpy.poly1d```. For finding reference manually read on.

Finding reference
-----------------

If ``<type>`` is not explicitly known it can be found out from ``objects.inv`` files found in URLs in intersphinx_mapping above.


To get and decode objects.inv for numpy::

    $ wget http://docs.scipy.org/doc/numpy/
    $ ipython
    
In python::

    import zlib
    with open("objects.inv","r") as f:
        inv_lines = f.readlines()
    lista = zlib.decompress(''.join(inv_lines[4:])).split('\n')
    with open('objects_numpy.inv','w') as f:
        for line in lista:
            f.write(line+'\n')

To find reference for ``numpy.poly1d`` serach for it in decoded file ``objects_numpy.inv``.

The line should include word ``class``

In documentation include ``:py:class:`numpy.poly1d```

For ``scipy.stats.norm``::

    find scipy.stats.norm -> data -> :py:data:`scipy.stats.norm`

Readthedocs.org
===============

In order for readthedocs.org to make documentation it needs to have certain packages accessible.

1. On readthedocs.org admin page check option ``Use virtualenv`` And ``Use system packages`` and in ``Requirements file`` put the name to the requirements file in repo (i.e. ``requirements.txt``).

2. Make ``readthedocs.py``  module and put it in ``docs`` folder with Mock class found `here  <http://read-the-docs.readthedocs.org/en/latest/faq.html#i-get-import-errors-on-libraries-that-depend-on-c-modules>`_

3. In documentation ``conf.py`` put the following lines to import mock class for certain modules that are not present in virtual environment. 
   ::
   
        if os.environ.get('READTHEDOCS', None) == 'True':
            sys.path.insert(0,'.')
            from readthedocs import *
            sys.path.pop(0)
