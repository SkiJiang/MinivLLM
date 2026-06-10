# Minimal setuptools metadata for installing the src-layout package locally.
from setuptools import setup, find_packages

setup(
    # Import name exposed by src/myvllm.
    name="myvllm",
    version="0.1.0",
    # Source files live under src/ instead of the repository root.
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # The project currently targets the exact Python version used by the local
    # development environment.
    python_requires="==3.11.14",
    install_requires=[
        # Core tensor/runtime dependency.  Other optional tools are used by
        # scripts and benchmarks but are not pinned here.
        "torch",
    ],
)
