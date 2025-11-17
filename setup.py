# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

from setuptools import find_packages, setup


def read_requirements():
    """Read requirements from core.txt file."""
    req_file = Path(__file__).parent / "requirements" / "core.txt"
    if req_file.exists():
        with open(req_file) as f:
            # Filter out comments and empty lines
            return [
                line.strip() for line in f
                if line.strip() and not line.strip().startswith('#')
            ]
    return []


# Collect configuration data files
data_files = []
for root, dirs, files in os.walk("configuration"):
    data_files.append((os.path.relpath(root, "configuration"),
                       [os.path.join(root, f) for f in files]))

setup(
    name="vllm-neuron",
    version="0.2.0",
    author="AWS Neuron team",
    license="Apache 2.0",
    description="vLLM Neuron backend plugin",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    packages=find_packages(exclude=("docs", "examples", "tests*", "csrc")),
    python_requires=">=3.10",
    install_requires=read_requirements(),
    entry_points={
        "vllm.platform_plugins": ["neuron = vllm_neuron:register"],
    },
    include_package_data=True,
)
