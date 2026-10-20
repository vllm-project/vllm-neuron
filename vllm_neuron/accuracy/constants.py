# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Constants for vLLM Neuron accuracy validation.
"""

# Tolerances tend to be tighter at smaller top_k values because the accuracy of
# more likely tokens is more important than less likely tokens.
DEFAULT_TOLERANCE_MAP = {
    "5": (1e-5, 0.011),
    "50": (1e-5, 0.02),
    "1000": (1e-5, 0.03),
    "all": (1e-5, 0.05),
}

DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE = 0.001
