# py-phasesep

[![Build status](https://github.com/zwicker-group/py-phasesep/workflows/build/badge.svg)](https://github.com/zwicker-group/py-phasesep/actions?query=workflow%3Abuild)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


`py-phasesep` provides python code for studying phase separation using numerical
simulations. In particular, it defines classes for general free energie densities and
for simulating extended Cahn-Hilliard equations. This package builds on the
[`py-pde` package](https://github.com/zwicker-group/py-pde), which is also developed in
our group. 


Installation
------------

Since this package is not public, it cannot be installed using `pip`. Instead the
repository needs to be cloned from github. The necessary python packages can be
installed using `pip`. To install the package together with the requirements, the
following commands can be used:

```bash
git clone https://github.com/zwicker-group/py-phasesep.git
pip install -r py-phasesep/requirements.txt
```


Documentation
-------------

The documentation for this package is not publicly available, but it can be generated
from the source code. To do this, additional requirements have to be installed and the
build script has to be called from the `docs` directory. The following commands, run
from the root directory of the repository, achieve this:

```bash
cd docs
pip install -r requirements.txt
make html
```

The main entry point to the documentation is then the webpage
`build/html/index.html`.


Running tests
-------------

The package comes with automated tests that reside in `tests` directories in the
respective python packages. The purpose of these tests is to ensure some basic
functionality of the package. Consequently, it is good practise to run the tests and fix
problems before commiting to the repository. There are a number of convenient scripts
collected in the root `tests` directory that can be used for this. In particular, there
is a `requirements.txt` for installing the necessary python components:
 
```bash
pip install -r tests/requirements.txt
```

The actual scripts in the `tests` directory servere different purposes:

* `tests_run.sh` runs all tests in sequential order. The script takes an
  optional argument that selects which tests are run: Only test files or methods
  that match the argument will be run.
* `tests_parallel.sh` runs all tests in parallel. Also takes a pattern argument.
* `tests_codestyle.sh` tests whether the code style is obeyed by all files. Problems
  in the code style should be resolved to achieve a uniform experience for everyone.
  Note that there is also a script `format_code.sh`, which enforces the code style using
  `isort` and `black`.
* `tests_types.sh` tests the type annotations in the python files. Type annotations are
  optional in python, but they can be helpful to spot subtle programming problems. 
