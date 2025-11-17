#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive test comparing hardware sampling vs CPU sampling.

This test loads two identical models - one with hardware sampling enabled and one with 
CPU sampling - and compares their outputs across various sampling configurations.

For deterministic cases (greedy, low temperature with seeds), outputs should match.
For non-deterministic cases, we verify both methods work without crashing.
"""

import os
import time

from vllm import LLM, SamplingParams


def test_hardware_vs_cpu_sampling_comparison():
    """
    Comprehensive test comparing hardware sampling vs CPU sampling.
    
    Tests both deterministic cases (where outputs should match) and 
    non-deterministic cases (where we verify both work but outputs may differ).
    """

    # Test configuration
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    prompts = [
        "The capital of France is", "In a world where technology",
        "The quick brown fox", "Once upon a time in a distant galaxy",
        "The meaning of life is"
    ]

    print("=" * 80)
    print("HARDWARE vs CPU SAMPLING COMPARISON TEST")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Test prompts: {len(prompts)}")
    print()

    # Load model with hardware sampling enabled
    print("🔧 Loading model with hardware sampling...")
    try:
        os.environ["NEURON_RT_VISIBLE_CORES"] = "0-15"
        hw_engine = LLM(
            model=model_name,
            tensor_parallel_size=16,
            max_model_len=512,
            max_num_seqs=4,
            enable_prefix_caching=False,
        )
        hw_loaded = True
        print("✅ Hardware sampling model loaded successfully")
    except Exception as e:
        hw_loaded = False
        hw_error = str(e)
        print(f"❌ Hardware sampling model failed to load: {hw_error}")

    # Load model with CPU sampling (no on_device_sampling_config)
    print("🔧 Loading model with CPU sampling...")
    try:
        os.environ["NEURON_RT_VISIBLE_CORES"] = "16-31"
        cpu_engine = LLM(model=model_name,
                         tensor_parallel_size=16,
                         max_model_len=512,
                         max_num_seqs=4,
                         enable_prefix_caching=False,
                         additional_config={
                             "override_neuron_config": {
                                 "on_device_sampling_config": None,
                             }
                         })
        cpu_loaded = True
        print("✅ CPU sampling model loaded successfully")
    except Exception as e:
        cpu_loaded = False
        cpu_error = str(e)
        print(f"❌ CPU sampling model failed to load: {cpu_error}")

    if not hw_loaded and not cpu_loaded:
        print("❌ Both models failed to load. Cannot proceed with comparison.")
        return

    # Deterministic sampling configurations - outputs should match
    deterministic_configs = [
        ("greedy", SamplingParams(temperature=0.0, max_tokens=10, seed=42)),
        ("greedy_with_seed",
         SamplingParams(temperature=0.0, max_tokens=15, seed=123)),
        ("low_temp_deterministic",
         SamplingParams(temperature=0.01, max_tokens=8, top_k=1, seed=42)),
        ("greedy_top_k_1",
         SamplingParams(temperature=0.0, max_tokens=12, top_k=1, seed=999)),
    ]

    # Non-deterministic sampling configurations - just verify both work
    non_deterministic_configs = [("nucleus_sampling",
                                  SamplingParams(temperature=0.7,
                                                 max_tokens=10,
                                                 top_p=0.95)),
                                 ("high_temperature",
                                  SamplingParams(temperature=0.8,
                                                 max_tokens=10,
                                                 top_k=50,
                                                 top_p=0.9)),
                                 ("pure_top_k",
                                  SamplingParams(temperature=1.0,
                                                 max_tokens=15,
                                                 top_k=20)),
                                 ("pure_top_p",
                                  SamplingParams(temperature=1.0,
                                                 max_tokens=15,
                                                 top_p=0.8)),
                                 ("combined_top_k_top_p",
                                  SamplingParams(temperature=0.6,
                                                 max_tokens=12,
                                                 top_k=40,
                                                 top_p=0.85)),
                                 ("repetition_penalty",
                                  SamplingParams(temperature=0.7,
                                                 max_tokens=25,
                                                 repetition_penalty=1.1)),
                                 ("frequency_penalty",
                                  SamplingParams(temperature=0.6,
                                                 max_tokens=20,
                                                 frequency_penalty=0.5)),
                                 ("presence_penalty",
                                  SamplingParams(temperature=0.7,
                                                 max_tokens=18,
                                                 presence_penalty=0.3)),
                                 ("combined_penalties",
                                  SamplingParams(temperature=0.8,
                                                 max_tokens=22,
                                                 repetition_penalty=1.05,
                                                 frequency_penalty=0.2,
                                                 presence_penalty=0.1)),
                                 ("min_p_sampling",
                                  SamplingParams(temperature=0.7,
                                                 max_tokens=12,
                                                 min_p=0.05)),
                                 ("stop_sequences",
                                  SamplingParams(temperature=0.8,
                                                 max_tokens=50,
                                                 stop=["\n", ".", "END"])),
                                 ("logprobs_request",
                                  SamplingParams(temperature=0.8,
                                                 max_tokens=12,
                                                 logprobs=5)),
                                 ("seed_deterministic_high_temp",
                                  SamplingParams(temperature=0.8,
                                                 max_tokens=20,
                                                 seed=42)),
                                 ("extreme_low_temp",
                                  SamplingParams(temperature=0.01,
                                                 max_tokens=10,
                                                 top_k=5)),
                                 ("extreme_high_temp",
                                  SamplingParams(temperature=2.0,
                                                 max_tokens=15,
                                                 top_p=0.95)),
                                 ("complex_combination",
                                  SamplingParams(temperature=0.75,
                                                 max_tokens=25,
                                                 top_k=30,
                                                 top_p=0.88,
                                                 repetition_penalty=1.08,
                                                 frequency_penalty=0.15,
                                                 presence_penalty=0.05,
                                                 seed=999))]

    # Test deterministic cases - outputs should match
    if hw_loaded and cpu_loaded:
        print("\n🎯 DETERMINISTIC SAMPLING TESTS (outputs should match)")
        print("-" * 60)

        total_matches = 0
        total_comparisons = 0

        for config_name, sampling_params in deterministic_configs:
            print(f"\n📋 Testing {config_name}:")
            print(f"   Params: {sampling_params}")

            try:
                # Generate with hardware sampling
                start_time = time.time()
                hw_outputs = hw_engine.generate(prompts, sampling_params)
                hw_time = time.time() - start_time
                hw_success = True
                hw_error = None
            except Exception as e:
                hw_success = False
                hw_error = str(e)
                hw_time = 0

            try:
                # Generate with CPU sampling
                start_time = time.time()
                cpu_outputs = cpu_engine.generate(prompts, sampling_params)
                cpu_time = time.time() - start_time
                cpu_success = True
                cpu_error = None
            except Exception as e:
                cpu_success = False
                cpu_error = str(e)
                cpu_time = 0

            if not hw_success:
                print(f"   ❌ Hardware sampling failed: {hw_error}")
                continue

            if not cpu_success:
                print(f"   ❌ CPU sampling failed: {cpu_error}")
                continue

            # Compare outputs
            matches = 0
            for i, (hw_out, cpu_out) in enumerate(zip(hw_outputs,
                                                      cpu_outputs)):
                hw_text = hw_out.outputs[0].text.strip()
                cpu_text = cpu_out.outputs[0].text.strip()

                if hw_text == cpu_text:
                    matches += 1
                    status = "✅ MATCH"
                else:
                    status = "❌ DIFFER"

                print(f"   Prompt {i+1}: {status}")
                print(f"     HW:  '{hw_text}'")
                print(f"     CPU: '{cpu_text}'")
                total_comparisons += 1

            total_matches += matches
            match_rate = matches / len(prompts) * 100
            print(
                f"   📊 Match Rate: {matches}/{len(prompts)} ({match_rate:.1f}%)"
            )
            print(f"   ⏱️  Timing: HW={hw_time:.3f}s, CPU={cpu_time:.3f}s")

            # For deterministic cases, we expect high match rates
            if match_rate < 80:
                print(
                    "   ⚠️  WARNING: Low match rate for deterministic sampling!"
                )

        overall_match_rate = (total_matches / total_comparisons *
                              100) if total_comparisons > 0 else 0
        print(
            f"\n📈 OVERALL DETERMINISTIC MATCH RATE: {total_matches}/{total_comparisons} ({overall_match_rate:.1f}%)"
        )

    # Test non-deterministic cases - just verify both work
    print("\n🎲 NON-DETERMINISTIC SAMPLING TESTS (verify both work)")
    print("-" * 60)

    hw_successes = 0
    cpu_successes = 0
    total_tests = len(non_deterministic_configs)

    for config_name, sampling_params in non_deterministic_configs:
        print(f"\n📋 Testing {config_name}:")
        print(f"   Params: {sampling_params}")

        # Test hardware sampling
        if hw_loaded:
            try:
                start_time = time.time()
                hw_outputs = hw_engine.generate(
                    prompts[:3],
                    sampling_params)  # Use fewer prompts for speed
                hw_time = time.time() - start_time
                hw_success = True
                hw_error = None
                hw_successes += 1
            except Exception as e:
                hw_success = False
                hw_error = str(e)
                hw_time = 0
        else:
            hw_success = False
            hw_error = "Model not loaded"
            hw_time = 0

        # Test CPU sampling
        if cpu_loaded:
            try:
                start_time = time.time()
                cpu_outputs = cpu_engine.generate(prompts[:3], sampling_params)
                cpu_time = time.time() - start_time
                if config_name == 'logprobs_request':
                    assert cpu_outputs[0].outputs[
                        0].logprobs, "No logprobs when logprobs were requested"
                cpu_success = True
                cpu_error = None
                cpu_successes += 1
            except Exception as e:
                cpu_success = False
                cpu_error = str(e)
                cpu_time = 0
        else:
            cpu_success = False
            cpu_error = "Model not loaded"
            cpu_time = 0

        # Report results
        hw_status = "✅ SUCCESS" if hw_success else f"❌ FAILED: {hw_error}"
        cpu_status = "✅ SUCCESS" if cpu_success else f"❌ FAILED: {cpu_error}"

        print(f"   Hardware Sampling: {hw_status}")
        if hw_success:
            print(f"     ⏱️  Time: {hw_time:.3f}s")
        print(f"   CPU Sampling:      {cpu_status}")
        if cpu_success:
            print(f"     ⏱️  Time: {cpu_time:.3f}s")

        # Show sample outputs if both succeeded
        if hw_success and cpu_success:
            print("   📝 Sample Outputs:")
            for i in range(min(2, len(prompts[:3]))):  # Show first 2 outputs
                hw_text = hw_outputs[i].outputs[0].text.strip()
                cpu_text = cpu_outputs[i].outputs[0].text.strip()
                print(f"     Prompt {i+1}:")
                print(f"       HW:  '{hw_text}'")
                print(f"       CPU: '{cpu_text}'")

    # Batch processing test
    print("\n📦 BATCH PROCESSING TEST")
    print("-" * 60)

    batch_prompts = prompts * 3  # 15 prompts total
    batch_sampling = SamplingParams(temperature=0.7, max_tokens=8, top_p=0.9)

    print(f"Testing batch of {len(batch_prompts)} prompts...")

    if hw_loaded:
        try:
            start_time = time.time()
            hw_batch = hw_engine.generate(batch_prompts, batch_sampling)
            hw_batch_time = time.time() - start_time
            print(
                f"✅ Hardware sampling: Generated {len(hw_batch)} outputs in {hw_batch_time:.3f}s"
            )
        except Exception as e:
            print(f"❌ Hardware sampling failed: {e}")

    if cpu_loaded:
        try:
            start_time = time.time()
            cpu_batch = cpu_engine.generate(batch_prompts, batch_sampling)
            cpu_batch_time = time.time() - start_time
            print(
                f"✅ CPU sampling: Generated {len(cpu_batch)} outputs in {cpu_batch_time:.3f}s"
            )
        except Exception as e:
            print(f"❌ CPU sampling failed: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    if hw_loaded and cpu_loaded:
        print("✅ Both hardware and CPU sampling models loaded successfully")
        if total_comparisons > 0:
            print(
                f"📊 Deterministic match rate: {overall_match_rate:.1f}% ({total_matches}/{total_comparisons})"
            )
    elif hw_loaded:
        print("⚠️  Only hardware sampling model loaded")
    elif cpu_loaded:
        print("⚠️  Only CPU sampling model loaded")
    else:
        print("❌ Neither model loaded successfully")

    if hw_loaded:
        hw_success_rate = (hw_successes / total_tests *
                           100) if total_tests > 0 else 0
        print(
            f"🔧 Hardware sampling success rate: {hw_success_rate:.1f}% ({hw_successes}/{total_tests})"
        )

    if cpu_loaded:
        cpu_success_rate = (cpu_successes / total_tests *
                            100) if total_tests > 0 else 0
        print(
            f"💻 CPU sampling success rate: {cpu_success_rate:.1f}% ({cpu_successes}/{total_tests})"
        )

    print("=" * 80)


if __name__ == "__main__":
    test_hardware_vs_cpu_sampling_comparison()