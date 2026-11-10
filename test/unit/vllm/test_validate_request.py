# SPDX-License-Identifier: Apache-2.0
"""`NeuronPlatform.validate_request` rejects requests as client errors (CPU).

`AsyncLLM` returns a `VLLMValidationError` to the client as a 400; it turns
any other exception, including `ValueError`, into an empty 500.
"""

import pytest
from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.inputs import PlaceholderRange
from vllm.sampling_params import StructuredOutputsParams

platform = pytest.importorskip("vllm_neuron.vllm.platform")


def test_structured_outputs_rejected_when_disabled(monkeypatch):
    monkeypatch.setattr(platform.NeuronPlatform, "_enable_structured_outputs", False)
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(json_object=True)
    )

    with pytest.raises(VLLMValidationError, match="enable_structured_outputs"):
        platform.NeuronPlatform.validate_request({}, params)


def test_multimodal_item_over_encoder_budget_rejected(monkeypatch):
    monkeypatch.setattr(platform.NeuronPlatform, "_max_embeds_per_image", 4)
    inputs = {"mm_placeholders": {"image": [PlaceholderRange(offset=0, length=8)]}}

    with pytest.raises(VLLMValidationError, match="8 embedding tokens"):
        platform.NeuronPlatform.validate_request(inputs, SamplingParams())
