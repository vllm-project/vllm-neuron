#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import logging
import os
from test.e2e.online.online_server_runner import (OnlineCfg,
                                                  run_online_integration)
from test.utils.fsx_utils.model_path import resolve_model_dir

import pytest

# ————— Logging setup —————
logger = logging.getLogger("sequence_parallel_test")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# ————— Constants —————
MODEL_NAME = "openlm-research/open_llama_3b"
SEQ_LEN = 6024  # Intentionally not divisible by TP_DEGREE to trigger validation
TP_DEGREE = 32
BATCH_SIZE = 1


def clear_server_logs(log_file="vllm_server_out.txt"):
    """Clear server logs from previous runs to avoid cross-test contamination"""
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
            logger.debug(f"Cleared previous server logs: {log_file}")
        except Exception as e:
            logger.warning(f"Could not clear log file {log_file}: {e}")


def check_server_logs_for_error(seq_len,
                                tp_degree,
                                log_file="vllm_server_out.txt"):
    """
    Check server logs for sequence parallel validation error.
    
    Args:
        seq_len: Expected sequence length in error message
        tp_degree: Expected TP degree in error message
        log_file: Path to server log file
        
    Returns:
        (bool, str): (error_found, log_content)
    """
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_content = f.read()
            error_msg = f"context length ({seq_len}) must be divisible by TP group size ({tp_degree})"
            if error_msg in log_content:
                logger.info(f"Found expected error in logs: {error_msg}")
                return True, log_content
    return False, ""


def verify_sequence_parallel_failure(title,
                                     seq_len,
                                     tp_degree,
                                     exception=None):
    """
    Verify that server failed with sequence parallel validation error.
    
    Args:
        title: Test title for logging
        seq_len: Expected sequence length in error message
        tp_degree: Expected TP degree in error message
        exception: Exception raised (if any), None if server started unexpectedly
        
    Raises:
        pytest.fail if error doesn't match expected pattern
    """
    found_error, log_content = check_server_logs_for_error(seq_len, tp_degree)

    expected_errors = [
        "context length", "divisible", "tp group size", "sequence parallel"
    ]

    # Check if error is in exception message
    error_in_exception = False
    if exception:
        error_str = str(exception).lower()
        error_in_exception = any(err in error_str for err in expected_errors)

    # Validation: error found in logs OR exception message
    if found_error or error_in_exception:
        if exception:
            logger.info(
                f"[{title}] ✓ Correctly failed with sequence parallel validation error: {str(exception)}"
            )
        else:
            logger.info(
                f"[{title}] ✓ Correctly failed with sequence parallel validation error (found in logs)"
            )
        return

    # Error not found - fail the test
    if exception is None:
        pytest.fail(
            f"[{title}] Expected server to fail with sequence parallel validation error, "
            f"but it started successfully.\n"
            f"Server logs excerpt:\n{log_content[:1000] if log_content else 'No logs found'}"
        )
    else:
        pytest.fail(
            f"[{title}] Got unexpected error: {str(exception)}\n"
            f"Server logs excerpt:\n{log_content[:1000] if log_content else 'No logs found'}"
        )


@pytest.mark.parametrize(
    "sequence_parallel_enabled, expect_failure",
    [(False, False), (True, True)],
    ids=["sequence_parallel_disabled", "sequence_parallel_enabled"],
)
def test_sequence_parallel(sequence_parallel_enabled, expect_failure):
    """
    Test sequence parallel functionality.
    
    - When disabled: Server should start normally
    - When enabled with non-divisible seq_len: Server should fail with validation error
    """
    title = f"{MODEL_NAME.split('/')[-1]}_seq_parallel_{'on' if sequence_parallel_enabled else 'off'}"
    logger.info(
        f"[{title}] Starting test (sequence_parallel_enabled={sequence_parallel_enabled})"
    )

    # Clear logs from previous runs to avoid cross-test contamination
    clear_server_logs()

    # Resolve model path
    model_path, _ = resolve_model_dir(MODEL_NAME)

    # Configure using OnlineCfg (which uses VllmServer internally)
    cfg = OnlineCfg(
        name=title,
        model=model_path,
        tp_degree=TP_DEGREE,
        batch_size=BATCH_SIZE,
        max_model_len=SEQ_LEN,
        max_tokens=100,
        accuracy_check=False,
        override_neuron_config={"sequence_parallel_enabled": True}
        if sequence_parallel_enabled else None,
    )

    if expect_failure:
        logger.info(
            f"[{title}] Expecting server to fail with validation error...")

        try:
            run_online_integration(cfg)
        except Exception as e:
            verify_sequence_parallel_failure(title,
                                             SEQ_LEN,
                                             TP_DEGREE,
                                             exception=e)
        else:
            verify_sequence_parallel_failure(title,
                                             SEQ_LEN,
                                             TP_DEGREE,
                                             exception=None)
    else:
        logger.info(f"[{title}] Starting server (should succeed)...")

        try:
            run_online_integration(cfg)
            logger.info(f"[{title}] ✓ Test passed successfully")
        except Exception as e:
            logger.error(f"[{title}] Unexpected failure: {e}")
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing with sequence_parallel disabled (should succeed)...")
    test_sequence_parallel(sequence_parallel_enabled=False,
                           expect_failure=False)

    print("\n" + "=" * 80 + "\n")

    print("Testing with sequence_parallel enabled (should fail)...")
    test_sequence_parallel(sequence_parallel_enabled=True, expect_failure=True)
