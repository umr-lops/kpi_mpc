# Dockerfile: micromamba environment with `kpimpc` installed
FROM mambaorg/micromamba:1.4.9 AS base

WORKDIR /app

# Create a micromamba env at a fixed prefix with Python and pip from conda-forge
# using an explicit prefix avoids surprises about where the env is installed.
RUN micromamba create -y -p /opt/conda/envs/kpimpc python=3.12 pip -c conda-forge \
    && micromamba clean --all --yes

# Ensure we have root privileges for package installation
USER root

# Install system packages required for editable installs (git for setuptools_scm)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Ensure subsequent RUNs use bash and the env's python in PATH
SHELL ["/bin/bash", "-lc"]
ENV CONDA_PREFIX=/opt/conda/envs/kpimpc
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Copy project source
COPY . /app

# ========== Development image ==========
FROM base AS development

# Install the package in editable mode for live development using the env's pip directly
RUN /opt/conda/envs/kpimpc/bin/pip install --upgrade pip setuptools wheel \
    && /opt/conda/envs/kpimpc/bin/pip install --no-cache-dir -e .

# Default dev command
CMD ["bash"]

# ========== Runtime image ==========
FROM base AS runtime

# Install the package for runtime (non-editable) using the env's pip directly
RUN /opt/conda/envs/kpimpc/bin/pip install --upgrade pip setuptools wheel \
    && /opt/conda/envs/kpimpc/bin/pip install --no-cache-dir . \
    && rm -rf /root/.cache/pip

# Clean source to avoid shipping sources (optional)
RUN rm -rf /app

# Default runtime command — adjust to your real entrypoint
CMD ["bash"]
