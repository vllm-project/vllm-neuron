# SPDX-License-Identifier: Apache-2.0
import os

from setuptools import find_packages, setup

data_files = []
for root, dirs, files in os.walk("configuration"):
    data_files.append((os.path.relpath(root, "configuration"),
                       [os.path.join(root, f) for f in files]))

setup(
    name="neuronx-vllm-plugin",
    version="0.1",
    author="AWS Neuron team",
    license="Apache 2.0",
    description=("vLLM Neuron backend plugin"),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(exclude=("docs", "examples", "tests*", "csrc")),
    python_requires=">=3.9",
    install_requires=[],
    entry_points={
        "vllm.platform_plugins": ["neuron = neuronx_vllm_plugin:register"],
    },
)
