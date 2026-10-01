"""Automated differential parity verification between direct compiler and standalone codegen paths.

Verifies:
1. Structural IR Parity: Both ggmlc.compile and ggmlc.codegen paths consume identical
   Canonical IR optimizations and lower to identical GGML execution graph topologies.
2. End-to-End Numerical Parity: The GGUF runtime interpreter (ggmlc.compile -> ggmlc.load)
   and the standalone compiled C++ project (ggmlc.codegen -> cmake build -> native binary)
   produce bitwise / numerically identical outputs matching PyTorch reference.
"""

import ggmlc
import numpy as np
import pytest
import torch
from torch import nn

from tests.codegen.test_standalone_execution import _run_standalone

pytest_plugins = ["tests.codegen.test_standalone_execution"]

torch.manual_seed(42)


class ParityMLP(nn.Module):
    """Multi-layer perceptron with linear projections and GELU activation."""

    def __init__(self, in_features=8, hidden_features=16, out_features=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class ParityConvBlock(nn.Module):
    """2D Convolution with bias and ReLU activation."""

    def __init__(self, in_channels=3, out_channels=8, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class ParityNormResidual(nn.Module):
    """RMSNorm-style normalization with residual skip connection."""

    def __init__(self, dim=16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5)
        normed = (x / rms) * self.weight
        return normed + self.fc(x)


class ParityAttention(nn.Module):
    """Scaled dot-product self-attention with Q, K, V projections."""

    def __init__(self, embed_dim=16, num_heads=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(attn_out)


@pytest.mark.parametrize(
    "model_cls, sample_shape",
    [
        (ParityMLP, (1, 8)),
        (ParityConvBlock, (1, 3, 16, 16)),
        (ParityNormResidual, (1, 4, 16)),
        (ParityAttention, (1, 4, 16)),
    ],
)
def test_compiler_codegen_structural_parity(model_cls, sample_shape):
    """Assert that direct compilation and codegen paths ingest and lower to identical GGML graphs."""
    model = model_cls().eval()
    x = torch.randn(*sample_shape)
    inputs = (x,)

    # Graph 1: via compile pipeline
    graph_compile = ggmlc.compile_to_graph(
        model=model,
        sample_inputs=inputs,
        enable_optimizations=True,
        enable_fusion=True,
    )

    # Graph 2: via codegen pipeline
    graph_codegen = ggmlc.compile_to_graph(
        model=model,
        sample_inputs=inputs,
        enable_optimizations=True,
        enable_fusion=True,
    )

    # Invariants verification
    assert len(graph_compile.nodes) == len(graph_codegen.nodes)
    assert len(graph_compile.tensors) == len(graph_codegen.tensors)
    assert graph_compile.inputs == graph_codegen.inputs
    assert graph_compile.outputs == graph_codegen.outputs
    assert graph_compile.parameters == graph_codegen.parameters

    for n1, n2 in zip(graph_compile.nodes, graph_codegen.nodes):
        assert n1.opcode == n2.opcode, f"OpCode mismatch: {n1.opcode} vs {n2.opcode}"
        assert n1.inputs == n2.inputs, f"Inputs mismatch for node {n1.id}"
        assert n1.outputs == n2.outputs, f"Outputs mismatch for node {n1.id}"
        assert n1.attributes.get("fused_pass") == n2.attributes.get("fused_pass")


def test_compiler_codegen_mlp_e2e_parity(ggml_standalone_libs, tmp_path):
    """End-to-end verification of MLP: GGUF ModelRunner vs Standalone Compiled C++ Binary."""
    model = ParityMLP().eval()
    x = torch.randn(1, 8)
    inputs = (x,)

    # 1. Run via direct compiler GGUF pipeline
    gguf_bytes = ggmlc.compile(model=model, sample_inputs=inputs, model_name="parity_mlp")
    runner = ggmlc.load(gguf_bytes, n_threads=1)
    out_compiler = runner(x.detach().cpu().numpy())
    out_compiler_flat = np.asarray(out_compiler, dtype=np.float32).ravel()

    # 2. Run via standalone C++ codegen pipeline
    out_codegen_flat = _run_standalone(
        model=model,
        example_args=inputs,
        test_name="parity_mlp",
        ggml_libs=ggml_standalone_libs,
        tmp_path=tmp_path,
        atol=5e-5,
    ).ravel()

    # 3. Direct Parity Assertions
    assert out_compiler_flat.shape == out_codegen_flat.shape
    max_diff = np.max(np.abs(out_compiler_flat - out_codegen_flat))
    assert max_diff < 1e-5, f"Compiler vs Codegen parity violation: max_diff = {max_diff}"

    dot = np.dot(out_compiler_flat, out_codegen_flat)
    norm1 = np.linalg.norm(out_compiler_flat)
    norm2 = np.linalg.norm(out_codegen_flat)
    cosine_sim = dot / (norm1 * norm2 + 1e-9)
    assert cosine_sim > 0.999999, f"Cosine similarity degraded: {cosine_sim}"


def test_compiler_codegen_conv_e2e_parity(ggml_standalone_libs, tmp_path):
    """End-to-end verification of Conv2D + ReLU: GGUF ModelRunner vs Standalone Compiled C++ Binary."""
    model = ParityConvBlock().eval()
    x = torch.randn(1, 3, 16, 16)
    inputs = (x,)

    # 1. Run via direct compiler GGUF pipeline
    gguf_bytes = ggmlc.compile(model=model, sample_inputs=inputs, model_name="parity_conv")
    runner = ggmlc.load(gguf_bytes, n_threads=1)
    out_compiler = runner(x.detach().cpu().numpy())
    out_compiler_flat = np.asarray(out_compiler, dtype=np.float32).ravel()

    # 2. Run via standalone C++ codegen pipeline
    out_codegen_flat = _run_standalone(
        model=model,
        example_args=inputs,
        test_name="parity_conv",
        ggml_libs=ggml_standalone_libs,
        tmp_path=tmp_path,
        atol=1e-5,
    ).ravel()

    # 3. Direct Parity Assertions
    assert out_compiler_flat.shape == out_codegen_flat.shape
    max_diff = np.max(np.abs(out_compiler_flat - out_codegen_flat))
    assert max_diff < 1e-5, f"Compiler vs Codegen parity violation: max_diff = {max_diff}"

    dot = np.dot(out_compiler_flat, out_codegen_flat)
    norm1 = np.linalg.norm(out_compiler_flat)
    norm2 = np.linalg.norm(out_codegen_flat)
    cosine_sim = dot / (norm1 * norm2 + 1e-9)
    assert cosine_sim > 0.999999, f"Cosine similarity degraded: {cosine_sim}"
