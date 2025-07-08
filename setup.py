"""
VllmNeuronPlugin packaging setup.

For more details on how to operate this file, check
https://w.amazon.com/index.php/Python/Brazil
"""

import os

from setuptools import setup

# Declare your non-python data files:
# Files underneath configuration/ will be copied into the build preserving the
# subdirectory structure if they exist.
data_files = []
for root, dirs, files in os.walk("configuration"):
    data_files.append(
        (os.path.relpath(root, "configuration"), [os.path.join(root, f) for f in files])
    )

setup(
    # include data files
    data_files=data_files,
)
