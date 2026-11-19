# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.5 per-layer NKI kernels, copied unchanged from sglang-neuron (see README.md).

The kernels read their switches from the environment when they are imported. The GDN prefill's widest matmul
defaults to 2048 moving columns, which a Trainium3 NeuronCore takes; a Trainium2 NeuronCore takes 512 (neuronx-cc:
"Matmul moving free dimension 2048 exceeds max 512 for nc_version gen3"), so this package sets 512 unless the
caller set a value.
"""

import os

os.environ.setdefault("QWEN35_GDNP_WIDE", "512")
