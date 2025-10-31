# SPDX-License-Identifier: Apache-2.0
"""
PyTest entrypoint for e2e online inference tests.
Starts the server, runs the client test via online_inference_runner.py,
and reports success/failure.
"""

import logging
from test.e2e.online.configs import (EAGLE_CONFIGS, INTEGRATION_CONFIGS,
                                     QWEN_CONFIG, TOOL_CALLING_CONFIGS)
from test.e2e.online.online_server_runner import run_online_integration

import pytest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("cfg",
                         INTEGRATION_CONFIGS,
                         ids=[c.name for c in INTEGRATION_CONFIGS])
def test_online_integration(cfg):
    """Parametrized integration tests"""
    logger.info("Running integration test with config: %s", cfg.name)
    run_online_integration(cfg)


@pytest.mark.timeout(900)
def test_qwen():
    """
    Test Qwen model with config override fix.
    
    This test verifies that ModelConfig overrides work correctly in spawned processes
    during online inference. Uses Qwen2.5-7B-Instruct which has 28 attention heads
    with tensor_parallel_size=32 to reproduce the exact scenario that was failing:
    "ValueError: Total number of attention heads (28) must be divisible by tensor parallel size (32)"
    
    If the config overrides are properly applied in spawned processes, the server
    should start successfully without validation errors.
    """
    logger.info("Running Qwen test with config: %s", QWEN_CONFIG.name)
    logger.info(
        "Testing model: %s with tp_degree=%d (28 attention heads should trigger override)",
        QWEN_CONFIG.model, QWEN_CONFIG.tp_degree)
    run_online_integration(QWEN_CONFIG)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("cfg",
                         EAGLE_CONFIGS,
                         ids=[c.name for c in EAGLE_CONFIGS])
def test_online_eagle(cfg):
    """Parametrized EAGLE speculative decoding tests"""
    logger.info("Running EAGLE test with config: %s", cfg.name)
    run_online_integration(cfg)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("cfg",
                         TOOL_CALLING_CONFIGS,
                         ids=[c.name for c in TOOL_CALLING_CONFIGS])
def test_online_tool_calling(cfg):
    """Parametrized tool calling tests"""
    logger.info("Running tool calling test with config: %s", cfg.name)
    run_online_integration(cfg)
