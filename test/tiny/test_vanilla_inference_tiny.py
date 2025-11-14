# SPDX-License-Identifier: Apache-2.0
"""
Tiny test for online inference - vanilla inference with TinyLlama.
"""

import logging
from test.e2e.online.configs import TINYTEST_CONFIG
from test.e2e.online.online_server_runner import run_online_integration

import pytest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


@pytest.mark.timeout(600)
def test_online_tinytest():
    """TinyLlama tiny test."""
    logger.info("Running tiny test with config: %s", TINYTEST_CONFIG.name)
    run_online_integration(TINYTEST_CONFIG)
