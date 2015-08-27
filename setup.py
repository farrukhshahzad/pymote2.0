import os
import sys
import distribute_setup
distribute_setup.use_setuptools()
from setuptools import setup, find_packages

# transfer profile_pymote for ipython into IPYTHONDIR
if 'install' in sys.argv or 'develop' in sys.argv:
    import shutil
    try:
        import IPython
        ipythondir = IPython.utils.path.get_ipython_dir()  # @UndefinedVariable
    except ImportError, AttributeError:  # @ReservedAssignment
        print("Pymote IPython configuration not installed. Install latest "
              "IPython and then copy the conf/ipython/profile_pymote/"
              "ipython_config.py manually to IPython config dir.")
    else:
        profiledir = os.path.join(ipythondir, 'profile_pymote')
        if not os.path.exists(ipythondir):
            os.makedirs(ipythondir)
        if not os.path.exists(profiledir):
            os.makedirs(profiledir)
        print ("copying ipython_config.py and ipython_notebook_config.py "
               "to "+profiledir)
        shutil.copy(os.path.join('pymote', 'conf', 'ipython',
                                 'ipython_config.py'), profiledir)
        shutil.copy(os.path.join('pymote', 'conf', 'ipython',
                                 'ipython_notebook_config.py'), profiledir)

sys.path.insert(0, 'pymote')
import release  # @UnresolvedImport
sys.path.pop(0)

setup(
    name=release.name,
    version=release.version,
    url=release.url,
    author=release.authors['Arbula'][0],
    author_email=release.authors['Arbula'][1],
    description=release.description,
    keywords=release.keywords,
    download_url=release.download_url,
    license=release.license,
    platforms=release.platforms,
    classifiers=release.classifiers,
    packages=find_packages(),
    #package_data = {
    #    '': ['*.bat'],
    #},
    exclude_package_data={'': ['README.rst']},
    install_requires=[
        'networkx',
        'numpy',
        'scipy',
        'pypng',
        'ipython',
        'matplotlib',
        #'PySide',
    ],
    long_description=open(os.path.join(os.path.dirname(__file__),
                                       'README.rst')).read(),

    entry_points={
        'pymote.algorithms': [],
        'console_scripts': [
            'ipymote = pymote.scripts.ipymote:start_ipymote',
        ],
        'gui_scripts': [
            'pymote-simgui = pymote.gui.simulationgui:main',
        ]
    },
    test_suite='nose.collector',
    tests_require=['nose'],

#    include_package_data=True,
#    zip_safe = False,
)
