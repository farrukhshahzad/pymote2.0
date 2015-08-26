name = 'Pymote'

major = 1
minor = 0
patch = 0

# Declare current release as a development release.
# Change to False before tagging a release; then change back.
dev = False

version = ''.join([str(major), '.', str(minor), '.', str(patch)])
if dev:
    version += '.dev'

description = ('A high-level Python library for simulation of distributed '
               'algorithms.')

long_description = \
"""
Pymote is a Python library for event based simulation and evaluation of
distributed algorithms.

"""
license = 'BSD'  # @ReservedAssignment
authors = {'Arbula': ('Damir Arbula', 'damir.arbula@gmail.com'),
           'Shahzad': ('Farrukh Shahzad', 'farrukhshahzad@kfupm.edu.sa'),
           }
#maintainer = "Pymote Developers"
#maintainer_email = "pymote-discuss@googlegroups.com"
url = 'https://github.com/darbula/pymote'
download_url = ''
platforms = ['Windows', 'Linux', 'Mac OSX']
keywords = ['Networks', 'Distributed algorithms']
classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: System :: Distributed Computing',
        'Topic :: System :: Networking',
        'Topic :: Scientific/Engineering :: Information Analysis']
