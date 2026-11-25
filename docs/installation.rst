Installation
============

With pip (Python 3.10 or later)::

    git clone https://github.com/umr-lops/kpi_mpc.git
    cd kpi_mpc
    pip install .

The documentation dependencies are an optional extra::

    pip install ".[docs]"
    sphinx-build -W -b html docs docs/_build/html

Apptainer image
---------------

The production image (used on the Ifremer HPC) is built from
``apptainer/kpimpc.def``, from the repository root. Its version comes from the
git tag::

    V=$(git describe --tags)
    apptainer build --fakeroot --build-arg KPIMPC_VERSION=$V \
        kpimpc_$V.sif apptainer/kpimpc.def
    apptainer inspect kpimpc_$V.sif    # version in the labels
    apptainer test kpimpc_$V.sif

The image contains the three commands (``kpinrcs``, ``kpihs``, ``kpipush``),
an ``sftp`` client for the push to the ECE, and ``ncat`` for an optional HTTP
CONNECT proxy.
