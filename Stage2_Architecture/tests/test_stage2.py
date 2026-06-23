"""
test_stage2.py -- Defense-Grade QA Suite for Stage 2 Hybrid Architecture
=========================================================================

This script performs rigorous validation of every component in the
Stage 2 Hybrid MER architecture, covering:

    1. Dimensionality & Tensor Routing (standard + edge-case batch sizes)
    2. Mathematical Integrity & SimAM (zero-variance NaN safety, unshared weights)
    3. Autograd & Backward Pass (gradient flow through entire computational graph)
    4. Device Agnosticism (CPU / CUDA transparent migration)
    5. Checkpoint Compatibility (save -> load -> bitwise output equality)
    6. Positional Encoding Correctness

Each test is logged via the Stage 1 dual-sink logger (console + rotating file)
so that every PASS/FAIL is auditable for the thesis committee.

Usage::
    # From Thesis3/ root:
    python -m Stage2_Architecture.tests.test_stage2

    # Or with pytest (auto-discovers test_* methods):
    pytest Stage2_Architecture/tests/test_stage2.py -v

Author  : Addhyan
Stage   : 2 -- Hybrid Neural Architecture (QA)
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup -- ensure Stage1 and Stage2 are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Stage1_DataPipeline"))

# -- Stage 1 Logger --------------------------------------------------------
from utils.logger import get_logger  # type: ignore[import]

# -- Stage 2 Modules Under Test --------------------------------------------
from Stage2_Architecture.models.simam import SimAM3D
from Stage2_Architecture.models.ststnet_backbone import STSTNetBackbone3D
from Stage2_Architecture.models.slstt_transformer import (
    SinusoidalPositionalEncoding,
    SLSTTTransformer,
)
from Stage2_Architecture.models.hybrid_model import HybridMERModel


# ---------------------------------------------------------------------------
# Logger instance -- writes to Stage2_Architecture/tests/logs/
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent / "logs"
log = get_logger(
    "test_stage2",
    log_dir=LOG_DIR,
    log_filename="test_stage2.log",
)


# ===================================================================
# Helper Utilities
# ===================================================================

def _build_model(**overrides) -> HybridMERModel:
    """Instantiate a HybridMERModel with sensible test defaults."""
    defaults = dict(
        num_classes=3,
        cnn_mid_channels=16,
        cnn_out_channels=32,
        cnn_dropout=0.0,          # Disable dropout for deterministic tests
        simam_lambda=1e-4,
        transformer_nhead=8,
        transformer_num_layers=2,  # Fewer layers for test speed
        transformer_dim_ff=128,
        transformer_dropout=0.0,
        pool_strategy="mean",
    )
    defaults.update(overrides)
    return HybridMERModel(**defaults)


def _dummy_input(batch: int = 2, device: str = "cpu") -> torch.Tensor:
    """Create a standard dummy input tensor [B, 3, 32, 224, 224]."""
    return torch.randn(batch, 3, 32, 224, 224, device=device)


# ===================================================================
# Section 1: Dimensionality & Tensor Routing Tests
# ===================================================================

class TestDimensionality(unittest.TestCase):
    """
    Validates that tensor shapes are correct at every stage of the
    forward pass -- from input through backbone, spatial pooling,
    transformer, and classification head.
    """

    @classmethod
    def setUpClass(cls) -> None:
        log.info("=" * 65)
        log.info("SECTION 1: Dimensionality & Tensor Routing Tests")
        log.info("=" * 65)
        cls.model = _build_model()
        cls.model.eval()

    # -- 1.1  Standard forward pass (B=2) --

    def test_standard_forward_shape(self) -> None:
        """
        TEST 1.1 -- Standard Forward Pass
        Input:    [2, 3, 32, 224, 224]
        Expected: [2, 3]
        Validates the canonical end-to-end shape transformation.
        """
        log.info("TEST 1.1: Standard forward pass [B=2, C=3, L=32, H=224, W=224]")

        x = _dummy_input(batch=2)
        with torch.no_grad():
            out = self.model(x)

        expected = (2, 3)
        self.assertEqual(
            out.shape, expected,
            f"Output shape {out.shape} != expected {expected}",
        )
        log.info("  [PASS] Output shape: %s", list(out.shape))

    # -- 1.2  Batch-size-1 edge case --

    def test_batch_size_one(self) -> None:
        """
        TEST 1.2 -- Batch Size 1 Edge Case
        Input:    [1, 3, 32, 224, 224]
        Expected: [1, 3]

        Why this matters:
            BatchNorm3d can behave unexpectedly with B=1 during
            training (it cannot compute batch statistics from a single
            sample). We test in eval() mode where running stats are
            used, which must succeed.
        """
        log.info("TEST 1.2: Batch size 1 edge case")

        x = _dummy_input(batch=1)
        with torch.no_grad():
            out = self.model(x)

        expected = (1, 3)
        self.assertEqual(
            out.shape, expected,
            f"Output shape {out.shape} != expected {expected}",
        )

        # Also verify no NaNs crept in
        self.assertFalse(
            torch.isnan(out).any().item(),
            "NaN detected in batch-size-1 output",
        )
        log.info("  [PASS] B=1 survived. Shape: %s, no NaNs.", list(out.shape))

    # -- 1.3  Backbone intermediate shape --

    def test_backbone_output_shape(self) -> None:
        """
        TEST 1.3 -- Backbone Intermediate Shape
        Input:    [2, 3, 32, 224, 224]
        Expected: [2, 96, 32, 112, 112]

        Validates the 3-stream concatenation produces the correct
        number of channels (32 x 3 = 96) and halved spatial dims.
        """
        log.info("TEST 1.3: Backbone intermediate shape verification")

        backbone = self.model.backbone
        backbone.eval()

        x = _dummy_input(batch=2)
        with torch.no_grad():
            feat = backbone(x)

        expected = (2, 96, 32, 112, 112)
        self.assertEqual(
            feat.shape, expected,
            f"Backbone output {feat.shape} != expected {expected}",
        )
        log.info("  [PASS] Backbone output: %s", list(feat.shape))

    # -- 1.4  Transformer intermediate shape --

    def test_transformer_output_shape(self) -> None:
        """
        TEST 1.4 -- Transformer Intermediate Shape
        Input:    [2, 32, 96]  (sequence of 32 time steps, 96-dim features)
        Expected: [2, 96]      (pooled temporal representation)
        """
        log.info("TEST 1.4: Transformer intermediate shape verification")

        transformer = self.model.transformer
        transformer.eval()

        x = torch.randn(2, 32, 96)
        with torch.no_grad():
            out = transformer(x)

        expected = (2, 96)
        self.assertEqual(
            out.shape, expected,
            f"Transformer output {out.shape} != expected {expected}",
        )
        log.info("  [PASS] Transformer output: %s", list(out.shape))

    # -- 1.5  CLS pooling strategy --

    def test_cls_pool_strategy(self) -> None:
        """
        TEST 1.5 -- CLS Token Pooling Strategy
        Verifies that the model also works with pool_strategy="cls",
        which prepends a learnable [CLS] token to the sequence.
        """
        log.info("TEST 1.5: CLS pooling strategy end-to-end")

        model_cls = _build_model(pool_strategy="cls")
        model_cls.eval()

        x = _dummy_input(batch=2)
        with torch.no_grad():
            out = model_cls(x)

        expected = (2, 3)
        self.assertEqual(out.shape, expected)
        log.info("  [PASS] CLS pooling output: %s", list(out.shape))


# ===================================================================
# Section 2: Mathematical Integrity & SimAM Tests
# ===================================================================

class TestMathematicalIntegrity(unittest.TestCase):
    """
    Validates the mathematical correctness and numerical stability
    of the SimAM attention module and the weight independence of
    the three CNN streams.
    """

    @classmethod
    def setUpClass(cls) -> None:
        log.info("")
        log.info("=" * 65)
        log.info("SECTION 2: Mathematical Integrity & SimAM Tests")
        log.info("=" * 65)

    # -- 2.1  SimAM: zero-variance input (all zeros) --

    def test_simam_zero_input_no_nan(self) -> None:
        """
        TEST 2.1 -- SimAM Zero-Variance Input (All Zeros)
        Input:  torch.zeros(2, 16, 32, 8, 8)

        When all activations are identical, (x - mu)^2 = 0 everywhere.
        The denominator becomes 4 * (0/n + lambda) = 4*lambda.
        Without lambda, this would be division by zero -> NaN.

        Energy = 0 / (4*lambda) + 0.5 = 0.5
        Sigmoid(0.5) ~ 0.6225
        Output = 0 * 0.6225 = 0   (all zeros, but finite)

        We assert: NO NaN, NO Inf in the output.
        """
        log.info("TEST 2.1: SimAM zero-input NaN safety check")

        simam = SimAM3D(e_lambda=1e-4)
        x_zeros = torch.zeros(2, 16, 32, 8, 8)

        out = simam(x_zeros)

        self.assertFalse(
            torch.isnan(out).any().item(),
            "NaN detected in SimAM output with all-zeros input!",
        )
        self.assertFalse(
            torch.isinf(out).any().item(),
            "Inf detected in SimAM output with all-zeros input!",
        )

        # Output should be all zeros (0 * sigmoid(0.5) = 0)
        self.assertTrue(
            torch.allclose(out, torch.zeros_like(out)),
            "Expected all-zero output when input is all zeros",
        )
        log.info("  [PASS] Zero input: no NaN, no Inf, output is all zeros.")

    # -- 2.2  SimAM: constant input (all ones) --

    def test_simam_constant_input_no_nan(self) -> None:
        """
        TEST 2.2 -- SimAM Constant Input (All Ones)
        Input:  torch.ones(2, 16, 32, 8, 8)

        When all activations are 1.0, (x - mu)^2 = 0 everywhere (mu=1).
        Energy = 0/(4*lambda) + 0.5 = 0.5 -> sigmoid(0.5) ~ 0.6225
        Output = 1.0 * 0.6225 ~ 0.6225

        Assert: no NaN, output ~ sigmoid(0.5) = 1/(1+e^{-0.5}).
        """
        log.info("TEST 2.2: SimAM constant-input (all ones) stability")

        simam = SimAM3D(e_lambda=1e-4)
        x_ones = torch.ones(2, 16, 32, 8, 8)

        out = simam(x_ones)

        self.assertFalse(
            torch.isnan(out).any().item(),
            "NaN detected with all-ones input!",
        )

        expected_val = torch.sigmoid(torch.tensor(0.5)).item()
        self.assertTrue(
            torch.allclose(out, torch.full_like(out, expected_val), atol=1e-5),
            f"Expected ~{expected_val:.4f} everywhere, got range "
            f"[{out.min().item():.6f}, {out.max().item():.6f}]",
        )
        log.info(
            "  [PASS] Constant input: output ~ sigmoid(0.5) = %.6f", expected_val
        )

    # -- 2.3  SimAM: parameter-free verification --

    def test_simam_is_parameter_free(self) -> None:
        """
        TEST 2.3 -- SimAM Has Zero Learnable Parameters
        The SimAM module must not introduce any learnable parameters.
        This is a core architectural invariant claimed in the thesis.
        """
        log.info("TEST 2.3: SimAM parameter-free assertion")

        simam = SimAM3D()
        num_params = sum(p.numel() for p in simam.parameters())

        self.assertEqual(
            num_params, 0,
            f"SimAM3D has {num_params} learnable parameters -- expected 0!",
        )
        log.info("  [PASS] SimAM3D has %d learnable parameters.", num_params)

    # -- 2.4  SimAM: shape preservation --

    def test_simam_preserves_shape(self) -> None:
        """
        TEST 2.4 -- SimAM Shape Preservation
        SimAM must return exactly the same shape as its input.
        Input:  [B, C, D, H, W] -> Output: [B, C, D, H, W]
        """
        log.info("TEST 2.4: SimAM shape preservation")

        simam = SimAM3D()
        shapes = [
            (1, 8, 16, 14, 14),
            (2, 16, 32, 56, 56),
            (4, 32, 32, 112, 112),
        ]

        for shape in shapes:
            x = torch.randn(*shape)
            out = simam(x)
            self.assertEqual(
                out.shape, x.shape,
                f"Shape mismatch: input {x.shape} -> output {out.shape}",
            )

        log.info("  [PASS] Shape preserved for %d test cases.", len(shapes))

    # -- 2.5  Unshared weights across streams --

    def test_unshared_stream_weights(self) -> None:
        """
        TEST 2.5 -- Unshared Weights Across CNN Streams
        The three CNN streams (stream_u, stream_v, stream_os) must
        have independent weight tensors -- they must NOT point to the
        same memory address (data_ptr).

        This verifies that modality-specific feature learning is
        possible (a key thesis contribution).
        """
        log.info("TEST 2.5: Unshared weight verification across 3 CNN streams")

        backbone = STSTNetBackbone3D()

        # Get conv1 weight data pointers from each stream
        ptr_u  = backbone.stream_u.conv1.weight.data_ptr()
        ptr_v  = backbone.stream_v.conv1.weight.data_ptr()
        ptr_os = backbone.stream_os.conv1.weight.data_ptr()

        self.assertNotEqual(
            ptr_u, ptr_v,
            "stream_u and stream_v share conv1 weight memory!",
        )
        self.assertNotEqual(
            ptr_u, ptr_os,
            "stream_u and stream_os share conv1 weight memory!",
        )
        self.assertNotEqual(
            ptr_v, ptr_os,
            "stream_v and stream_os share conv1 weight memory!",
        )

        log.info(
            "  [PASS] Weight pointers are distinct:\n"
            "           stream_u.conv1:  0x%016x\n"
            "           stream_v.conv1:  0x%016x\n"
            "           stream_os.conv1: 0x%016x",
            ptr_u, ptr_v, ptr_os,
        )

    # -- 2.6  Unshared weights -- value divergence after update --

    def test_unshared_weights_diverge_after_step(self) -> None:
        """
        TEST 2.6 -- Streams Diverge After One Gradient Step
        Even if weights start similar (due to same init distribution),
        a single backward pass with per-stream input should produce
        different gradients -> different weights after optimizer.step().

        Procedure:
            1. Create backbone, feed different data to each stream
            2. Compute loss, backward, optimizer.step()
            3. Assert conv1 weights of stream_u != stream_v != stream_os
        """
        log.info("TEST 2.6: Weight divergence after one gradient step")

        backbone = STSTNetBackbone3D(dropout_p=0.0)
        backbone.train()
        optimizer = torch.optim.SGD(backbone.parameters(), lr=0.01)

        # Craft input where each modality channel has different statistics
        x = torch.randn(2, 3, 32, 56, 56)
        x[:, 0, :, :, :] *= 10.0    # Scale u channel
        x[:, 1, :, :, :] *= 0.01    # Shrink v channel
        # os channel stays at normal scale

        out = backbone(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        w_u  = backbone.stream_u.conv1.weight.data.clone()
        w_v  = backbone.stream_v.conv1.weight.data.clone()
        w_os = backbone.stream_os.conv1.weight.data.clone()

        self.assertFalse(
            torch.allclose(w_u, w_v, atol=1e-7),
            "stream_u and stream_v weights are identical after gradient step!",
        )
        self.assertFalse(
            torch.allclose(w_u, w_os, atol=1e-7),
            "stream_u and stream_os weights are identical after gradient step!",
        )
        log.info("  [PASS] Weights diverged after one gradient step.")


# ===================================================================
# Section 3: Autograd & Backward Pass Tests
# ===================================================================

class TestAutogradBackward(unittest.TestCase):
    """
    Validates that the computational graph is fully connected from
    output logits all the way back to every learnable parameter.
    A broken gradient chain would silently prevent training.
    """

    @classmethod
    def setUpClass(cls) -> None:
        log.info("")
        log.info("=" * 65)
        log.info("SECTION 3: Autograd & Backward Pass Tests")
        log.info("=" * 65)
        cls.model = _build_model(cnn_dropout=0.0, transformer_dropout=0.0)

    # -- 3.1  Full backward pass completes without error --

    def test_backward_pass_completes(self) -> None:
        """
        TEST 3.1 -- Backward Pass Completes
        Perform a forward + backward pass using a dummy loss.
        This must not raise any RuntimeError (e.g., "element 0 of
        tensors does not require grad").
        """
        log.info("TEST 3.1: Full backward pass completion")

        self.model.train()
        self.model.zero_grad()

        x = _dummy_input(batch=2)
        logits = self.model(x)
        loss = logits.sum()

        # This should NOT raise
        try:
            loss.backward()
            passed = True
        except RuntimeError as e:
            passed = False
            log.error("  [FAIL] backward() raised: %s", e)

        self.assertTrue(passed, "backward() raised a RuntimeError")
        log.info("  [PASS] backward() completed without error.")

    # -- 3.2  All trainable parameters receive gradients --

    def test_all_parameters_receive_gradients(self) -> None:
        """
        TEST 3.2 -- All Parameters Receive Gradients
        After backward(), every parameter with requires_grad=True
        must have a non-None .grad tensor. A None gradient means
        that parameter is disconnected from the computation graph
        and would never be updated by the optimizer.

        We categorise parameters by component for clear diagnostics.
        """
        log.info("TEST 3.2: Gradient flow to all trainable parameters")

        self.model.train()
        self.model.zero_grad()

        x = _dummy_input(batch=2)
        logits = self.model(x)
        loss = logits.sum()
        loss.backward()

        no_grad_params: List[str] = []
        total_params = 0
        graded_params = 0

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is None:
                    no_grad_params.append(name)
                else:
                    graded_params += 1

        if no_grad_params:
            log.error(
                "  [FAIL] %d parameters have None gradients:", len(no_grad_params)
            )
            for pname in no_grad_params:
                log.error("           - %s", pname)

        self.assertEqual(
            len(no_grad_params), 0,
            f"{len(no_grad_params)} parameter(s) have no gradient: {no_grad_params}",
        )
        log.info(
            "  [PASS] %d/%d trainable parameters received gradients.",
            graded_params, total_params,
        )

    # -- 3.3  Gradient magnitudes are finite (no NaN/Inf) --

    def test_gradients_are_finite(self) -> None:
        """
        TEST 3.3 -- Gradient Magnitudes Are Finite
        After backward(), verify that no gradient tensor contains
        NaN or Inf values, which would indicate numerical instability
        (e.g., exploding gradients or division-by-zero in SimAM).
        """
        log.info("TEST 3.3: Gradient finiteness check (no NaN/Inf)")

        self.model.train()
        self.model.zero_grad()

        x = _dummy_input(batch=2)
        logits = self.model(x)
        loss = logits.sum()
        loss.backward()

        bad_grads: List[str] = []

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    bad_grads.append(name)

        self.assertEqual(
            len(bad_grads), 0,
            f"NaN/Inf gradients found in: {bad_grads}",
        )
        log.info("  [PASS] All gradients are finite.")

    # -- 3.4  SimAM does not block gradient flow --

    def test_simam_gradient_passthrough(self) -> None:
        """
        TEST 3.4 -- SimAM Gradient Passthrough
        SimAM is parameter-free, but it must NOT detach the gradient
        tape. We verify that gradients flow through it by:
            1. Creating a tensor with requires_grad=True
            2. Passing through SimAM
            3. Calling backward on the output
            4. Asserting input.grad is not None
        """
        log.info("TEST 3.4: SimAM gradient passthrough verification")

        simam = SimAM3D()
        x = torch.randn(1, 8, 16, 14, 14, requires_grad=True)

        out = simam(x)
        out.sum().backward()

        self.assertIsNotNone(
            x.grad,
            "SimAM blocked gradient flow -- input.grad is None!",
        )
        self.assertFalse(
            torch.isnan(x.grad).any().item(),
            "NaN in gradient flowing through SimAM!",
        )
        log.info(
            "  [PASS] Gradients flow through SimAM. Input grad norm: %.6f",
            x.grad.norm().item(),
        )

    # -- 3.5  CrossEntropy loss backward (realistic training) --

    def test_crossentropy_backward(self) -> None:
        """
        TEST 3.5 -- Realistic CrossEntropy Loss Backward Pass
        Simulates a real training step with CrossEntropyLoss and
        class targets. This catches dtype mismatches and shape
        incompatibilities that dummy loss.sum() might miss.
        """
        log.info("TEST 3.5: Realistic CrossEntropy loss backward")

        self.model.train()
        self.model.zero_grad()

        x = _dummy_input(batch=4)
        targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)

        logits = self.model(x)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, targets)

        loss.backward()

        # Verify loss is a scalar and finite
        self.assertEqual(loss.dim(), 0, "Loss is not a scalar!")
        self.assertFalse(torch.isnan(loss).item(), "Loss is NaN!")
        self.assertFalse(torch.isinf(loss).item(), "Loss is Inf!")

        log.info("  [PASS] CrossEntropy loss = %.4f, backward OK.", loss.item())


# ===================================================================
# Section 4: Device Agnosticism Tests
# ===================================================================

class TestDeviceAgnosticism(unittest.TestCase):
    """
    Validates that the model works on both CPU and (if available) CUDA,
    with special attention to registered buffers (e.g., positional
    encoding) that might not auto-migrate to GPU.
    """

    @classmethod
    def setUpClass(cls) -> None:
        log.info("")
        log.info("=" * 65)
        log.info("SECTION 4: Device Agnosticism Tests")
        log.info("=" * 65)
        cls.has_cuda = torch.cuda.is_available()
        if cls.has_cuda:
            log.info("  CUDA available: %s", torch.cuda.get_device_name(0))
        else:
            log.info("  CUDA not available -- GPU tests will be skipped.")

    # -- 4.1  CPU forward pass --

    def test_cpu_forward(self) -> None:
        """
        TEST 4.1 -- CPU Forward Pass
        Baseline: model on CPU, data on CPU. Must always work.
        """
        log.info("TEST 4.1: CPU forward pass")

        model = _build_model()
        model.eval()
        x = _dummy_input(batch=2, device="cpu")

        with torch.no_grad():
            out = model(x)

        self.assertEqual(out.device.type, "cpu")
        self.assertEqual(out.shape, (2, 3))
        log.info("  [PASS] CPU forward: %s on %s", list(out.shape), out.device)

    # -- 4.2  CUDA forward pass --

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_forward(self) -> None:
        """
        TEST 4.2 -- CUDA Forward Pass
        Moves model and data to GPU. This catches the common bug where
        registered buffers (like positional encoding) remain on CPU
        while the rest of the model is on GPU, causing a device
        mismatch RuntimeError.
        """
        log.info("TEST 4.2: CUDA forward pass")

        device = torch.device("cuda")
        model = _build_model().to(device)
        model.eval()

        x = _dummy_input(batch=2, device="cuda")

        with torch.no_grad():
            out = model(x)

        self.assertEqual(out.device.type, "cuda")
        self.assertEqual(out.shape, (2, 3))
        self.assertFalse(torch.isnan(out).any().item())
        log.info("  [PASS] CUDA forward: %s on %s", list(out.shape), out.device)

    # -- 4.3  CUDA backward pass --

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_backward(self) -> None:
        """
        TEST 4.3 -- CUDA Backward Pass
        Full forward + backward on GPU to catch device placement bugs
        in all gradient paths.
        """
        log.info("TEST 4.3: CUDA backward pass")

        device = torch.device("cuda")
        model = _build_model().to(device)
        model.train()

        x = _dummy_input(batch=2, device="cuda")
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        # Verify at least one gradient exists on the correct device
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.assertEqual(
                    param.grad.device.type, "cuda",
                    f"Gradient for {name} is on {param.grad.device}, expected cuda!",
                )
                break

        log.info("  [PASS] CUDA backward completed, gradients on correct device.")

    # -- 4.4  Positional encoding buffer device --

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_pe_buffer_migrates_to_cuda(self) -> None:
        """
        TEST 4.4 -- Positional Encoding Buffer Device Migration
        The sinusoidal PE is a registered buffer (not a parameter).
        When .to(device) is called on the model, the buffer must
        also migrate. If it doesn't, we get:
            RuntimeError: Expected all tensors to be on the same device

        This is one of the most common PyTorch deployment bugs.
        """
        log.info("TEST 4.4: PE buffer device migration check")

        device = torch.device("cuda")
        model = _build_model().to(device)

        # Navigate to the PE buffer
        pe_buffer = model.transformer.pos_encoder.pe

        self.assertEqual(
            pe_buffer.device.type, "cuda",
            f"PE buffer is on {pe_buffer.device}, expected cuda! "
            f"This would cause a device mismatch during forward pass.",
        )
        log.info("  [PASS] PE buffer on %s", pe_buffer.device)


# ===================================================================
# Section 5: Checkpoint Compatibility (Save / Load State)
# ===================================================================

class TestCheckpointCompatibility(unittest.TestCase):
    """
    Validates that the model state_dict can be saved to disk,
    loaded into a fresh instance, and produce bitwise-identical
    outputs -- proving that checkpointing will work in Stage 3.
    """

    @classmethod
    def setUpClass(cls) -> None:
        log.info("")
        log.info("=" * 65)
        log.info("SECTION 5: Checkpoint Compatibility Tests")
        log.info("=" * 65)

    # -- 5.1  Save -> Load -> Identical Output --

    def test_save_load_output_equality(self) -> None:
        """
        TEST 5.1 -- Save -> Load -> Output Equality
        Procedure:
            1. Create model_A, pass dummy input, record output.
            2. Save model_A.state_dict() to a temp file.
            3. Create model_B (fresh, different random init).
            4. Load state_dict from file into model_B.
            5. Pass SAME dummy input through model_B.
            6. Assert output_A == output_B (bitwise, not just close).

        This proves checkpointing integrity for Stage 3 training.
        """
        log.info("TEST 5.1: Save -> Load -> Identical output verification")

        # -- Step 1: Create and evaluate model_A --
        model_a = _build_model()
        model_a.eval()

        # Use a fixed seed for the dummy input
        torch.manual_seed(42)
        x = _dummy_input(batch=2)

        with torch.no_grad():
            output_a = model_a(x)

        # -- Step 2: Save state_dict to temp file --
        tmp_dir = Path(PROJECT_ROOT / "Stage2_Architecture" / "tests" / "_tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = tmp_dir / "test_checkpoint.pth"

        try:
            torch.save(model_a.state_dict(), str(ckpt_path))
            log.info("  Saved checkpoint: %s", ckpt_path)
            self.assertTrue(
                ckpt_path.exists(),
                f"Checkpoint file was not created at {ckpt_path}",
            )

            # -- Step 3: Create fresh model_B --
            model_b = _build_model()
            model_b.eval()

            # -- Step 4: Load state_dict into model_B --
            state_dict = torch.load(str(ckpt_path), weights_only=True)
            model_b.load_state_dict(state_dict)
            model_b.eval()

            # -- Step 5: Forward pass with same input --
            with torch.no_grad():
                output_b_after = model_b(x)

            # -- Step 6: Assert bitwise equality --
            self.assertTrue(
                torch.equal(output_a, output_b_after),
                f"Output mismatch after checkpoint load!\n"
                f"  model_A output: {output_a}\n"
                f"  model_B output: {output_b_after}\n"
                f"  Max diff: {(output_a - output_b_after).abs().max().item():.2e}",
            )
            log.info(
                "  [PASS] Outputs are BITWISE IDENTICAL after save/load.\n"
                "           model_A: %s\n"
                "           model_B: %s",
                output_a.tolist(), output_b_after.tolist(),
            )

        finally:
            # -- Cleanup: remove temp checkpoint --
            if ckpt_path.exists():
                ckpt_path.unlink()
                log.info("  Cleaned up: %s", ckpt_path)
            if tmp_dir.exists() and not any(tmp_dir.iterdir()):
                tmp_dir.rmdir()
                log.info("  Cleaned up: %s", tmp_dir)

    # -- 5.2  State dict key completeness --

    def test_state_dict_key_completeness(self) -> None:
        """
        TEST 5.2 -- State Dict Key Completeness
        Verify that the state_dict contains entries for all expected
        components:
            - backbone.stream_u.*
            - backbone.stream_v.*
            - backbone.stream_os.*
            - transformer.*
            - classifier.*

        Missing keys would cause silent partial loading.
        """
        log.info("TEST 5.2: State dict key completeness check")

        model = _build_model()
        sd = model.state_dict()

        required_prefixes = [
            "backbone.stream_u.",
            "backbone.stream_v.",
            "backbone.stream_os.",
            "transformer.pos_encoder.",
            "transformer.transformer_encoder.",
            "classifier.",
        ]

        for prefix in required_prefixes:
            matching_keys = [k for k in sd.keys() if k.startswith(prefix)]
            self.assertGreater(
                len(matching_keys), 0,
                f"No state_dict keys found with prefix '{prefix}'!",
            )

        log.info(
            "  [PASS] State dict has %d keys covering all %d component prefixes.",
            len(sd), len(required_prefixes),
        )

    # -- 5.3  In-memory buffer save/load (no disk I/O) --

    def test_in_memory_checkpoint(self) -> None:
        """
        TEST 5.3 -- In-Memory Checkpoint (BytesIO)
        Save/load via io.BytesIO to avoid filesystem dependencies.
        Verifies that state_dict serialisation works with arbitrary
        I/O backends (e.g., cloud storage in production).
        """
        log.info("TEST 5.3: In-memory checkpoint via BytesIO")

        model_a = _build_model()
        model_a.eval()

        torch.manual_seed(123)
        x = _dummy_input(batch=2)

        with torch.no_grad():
            out_a = model_a(x)

        # Save to BytesIO buffer
        buffer = io.BytesIO()
        torch.save(model_a.state_dict(), buffer)

        # Load from buffer into fresh model
        buffer.seek(0)
        model_b = _build_model()
        model_b.load_state_dict(torch.load(buffer, weights_only=True))
        model_b.eval()

        with torch.no_grad():
            out_b = model_b(x)

        self.assertTrue(
            torch.equal(out_a, out_b),
            "In-memory checkpoint output mismatch!",
        )
        log.info("  [PASS] In-memory save/load produces identical outputs.")


# ===================================================================
# Section 6: Positional Encoding Correctness
# ===================================================================

class TestPositionalEncoding(unittest.TestCase):
    """
    Validates the mathematical correctness of the sinusoidal
    positional encoding module.
    """

    @classmethod
    def setUpClass(cls) -> None:
        log.info("")
        log.info("=" * 65)
        log.info("SECTION 6: Positional Encoding Correctness Tests")
        log.info("=" * 65)

    # -- 6.1  PE is deterministic (not random) --

    def test_pe_is_deterministic(self) -> None:
        """
        TEST 6.1 -- PE Buffer Is Deterministic
        Two independently constructed PE modules with the same d_model
        and max_len must produce identical encoding matrices.
        """
        log.info("TEST 6.1: Positional encoding determinism")

        pe1 = SinusoidalPositionalEncoding(d_model=96, max_len=64, dropout_p=0.0)
        pe2 = SinusoidalPositionalEncoding(d_model=96, max_len=64, dropout_p=0.0)

        self.assertTrue(
            torch.equal(pe1.pe, pe2.pe),
            "Two PE modules with same config produced different encodings!",
        )
        log.info("  [PASS] PE is deterministic across instances.")

    # -- 6.2  PE values are bounded --

    def test_pe_values_bounded(self) -> None:
        """
        TEST 6.2 -- PE Values Are Bounded In [-1, 1]
        Sinusoidal PE uses sin/cos, so all values must be in [-1, 1].
        """
        log.info("TEST 6.2: PE value bounds check")

        pe = SinusoidalPositionalEncoding(d_model=96, max_len=128)

        self.assertGreaterEqual(pe.pe.min().item(), -1.0)
        self.assertLessEqual(pe.pe.max().item(), 1.0)
        log.info(
            "  [PASS] PE range: [%.4f, %.4f]",
            pe.pe.min().item(), pe.pe.max().item(),
        )

    # -- 6.3  PE does not have learnable parameters --

    def test_pe_has_no_parameters(self) -> None:
        """
        TEST 6.3 -- PE Has No Learnable Parameters
        The sinusoidal PE is a fixed buffer, not a learned embedding.
        """
        log.info("TEST 6.3: PE parameter count")

        pe = SinusoidalPositionalEncoding(d_model=96)
        num_params = sum(p.numel() for p in pe.parameters())

        # Only the Dropout layer has no params, pe is a buffer
        self.assertEqual(num_params, 0, f"PE has {num_params} params, expected 0")
        log.info("  [PASS] PE has %d learnable parameters.", num_params)


# ===================================================================
# Test Runner -- Summary Report
# ===================================================================

def _run_all_tests() -> None:
    """Execute all test sections and produce a summary report."""

    log.info("")
    log.info("+" + "=" * 63 + "+")
    log.info("|   STAGE 2 -- DEFENSE-GRADE QA SUITE                            |")
    log.info("|   HybridMERModel Architecture Validation                       |")
    log.info("+" + "=" * 63 + "+")
    log.info("")

    # Build the test suite in logical order
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestDimensionality,
        TestMathematicalIntegrity,
        TestAutogradBackward,
        TestDeviceAgnosticism,
        TestCheckpointCompatibility,
        TestPositionalEncoding,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    # Run with verbosity=2 for detailed test names
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # -- Summary --
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = total - failures - errors - skipped

    log.info("")
    log.info("+" + "=" * 63 + "+")
    log.info(
        "|   RESULTS: %d/%d PASSED  |  %d Failed  |  %d Errors  |  %d Skipped   |",
        passed, total, failures, errors, skipped,
    )
    log.info("+" + "=" * 63 + "+")

    if failures or errors:
        log.error("  [FAIL] SUITE FAILED -- See details above.")
        for test, traceback in result.failures + result.errors:
            log.error("  FAILED: %s", test)
    else:
        log.info("  [PASS] ALL TESTS PASSED -- Architecture validated for Stage 3.")


if __name__ == "__main__":
    _run_all_tests()
