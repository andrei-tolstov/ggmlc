from __future__ import annotations

import re
from pathlib import Path

from ggmlc.dialect.ggml.lowering import GGMLExecutionGraph, GGMLOpDef
from ggmlc.dialect.ggml.ops import GGMLOpCode
from ggmlc.ir.shape import AddDim, Dim, FloorDivDim, MulDim, StaticDim, SubDim, SymbolDim


def _sanitize_ident(name: str) -> str:
    """Sanitizes a tensor name into a valid C++ identifier."""
    ident = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if ident and ident[0].isdigit():
        ident = "t_" + ident
    return ident or "t_unnamed"


def _dim_to_cpp_expr(d: Dim | int | None) -> str:
    """Converts a Dimension expression into a C++ expression string."""
    if d is None:
        return "1"
    if isinstance(d, int):
        return str(d)
    if isinstance(d, StaticDim):
        return str(d.value)
    if isinstance(d, SymbolDim):
        return f'(symbols.count("{d.name}") ? symbols.at("{d.name}") : 1)'
    if isinstance(d, AddDim):
        return f"({_dim_to_cpp_expr(d.left)} + {_dim_to_cpp_expr(d.right)})"
    if isinstance(d, SubDim):
        return f"({_dim_to_cpp_expr(d.left)} - {_dim_to_cpp_expr(d.right)})"
    if isinstance(d, MulDim):
        return f"({_dim_to_cpp_expr(d.left)} * {_dim_to_cpp_expr(d.right)})"
    if isinstance(d, FloorDivDim):
        return f"({_dim_to_cpp_expr(d.left)} / {_dim_to_cpp_expr(d.right)})"
    return "1"


class GGMLCCppCodeGenerator:
    """Generates a standalone, human-readable C++ project from a GGMLExecutionGraph."""

    def __init__(self, graph: GGMLExecutionGraph, model_name: str = "model"):
        self.graph = graph
        self.model_name = _sanitize_ident(model_name)

    def generate_header(self) -> str:
        """Generates model.h containing weights struct and build_graph function."""
        in_descs = [
            f'"{self.graph.tensors[i].name}" [{"x".join(_dim_to_cpp_expr(d) for d in self.graph.tensors[i].ne)}]'
            for i in self.graph.inputs
            if i in self.graph.tensors
        ]
        out_descs = [
            f'"{self.graph.tensors[i].name}" [{"x".join(_dim_to_cpp_expr(d) for d in self.graph.tensors[i].ne)}]'
            for i in self.graph.outputs
            if i in self.graph.tensors
        ]
        in_summary = ", ".join(in_descs) if in_descs else "none"
        out_summary = ", ".join(out_descs) if out_descs else "none"

        lines: list[str] = [
            "// ============================================================================",
            f"// Model Architecture & Execution Graph: {self.model_name}",
            "// Automatically transpiled by ggmlc. Native execution across CPU, CUDA, Metal.",
            "//",
            "// Topology Summary:",
            f"//   - Declared Inputs ({len(self.graph.inputs)}): {in_summary}",
            f"//   - Model Parameters ({len(self.graph.parameters)} tensors)",
            f"//   - Declared Outputs ({len(self.graph.outputs)}): {out_summary}",
            f"//   - Computational Operations: {len(self.graph.nodes)} execution graph nodes",
            "// ============================================================================",
            "#pragma once",
            "",
            "#include <string>",
            "#include <vector>",
            "#include <unordered_map>",
            "#include <memory>",
            "#include <cmath>",
            "#include <cfloat>",
            "#include <algorithm>",
            "#include <utility>",
            "#include <fstream>",
            "#include <iostream>",
            '#include "ggml.h"',
            '#include "ggml-alloc.h"',
            '#include "ggml-backend.h"',
            '#include "ggml-cpu.h"',
            "#if defined(GGML_USE_CUDA)",
            '#include "ggml-cuda.h"',
            "#endif",
            "#if defined(GGML_USE_METAL)",
            '#include "ggml-metal.h"',
            "#endif",
            '#include "gguf.h"',
            '#include "ggmlc/stdlib_kernels.h"',
            "",
            f"namespace {self.model_name} {{",
            "",
            "// ----------------------------------------------------------------------------",
            "// Broadcast helper for elementwise ops.",
            "// ggml binary ops need `b` repeatable to `a`; this aligns the pair first.",
            "// ----------------------------------------------------------------------------",
            "inline std::pair<struct ggml_tensor*, struct ggml_tensor*> match_broadcast(",
            "    struct ggml_context* ctx,",
            "    struct ggml_tensor* a,",
            "    struct ggml_tensor* b) {",
            "    if (!a || !b) return {a, b};",
            "    if (ggml_are_same_shape(a, b)) return {a, b};",
            "    if (a->ne[1] == 1 && a->ne[2] == 1 && a->ne[3] == 1 && a->ne[0] == b->ne[2]) {",
            "        if (!ggml_is_contiguous(a)) a = ggml_cont(ctx, a);",
            "        a = ggml_reshape_4d(ctx, a, 1, 1, a->ne[0], 1);",
            "    }",
            "    if (b->ne[1] == 1 && b->ne[2] == 1 && b->ne[3] == 1 && b->ne[0] == a->ne[2]) {",
            "        if (!ggml_is_contiguous(b)) b = ggml_cont(ctx, b);",
            "        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);",
            "    }",
            "    if (ggml_can_repeat(b, a)) return {a, b};",
            "    if (ggml_can_repeat(a, b)) {",
            "        if (!ggml_is_contiguous(a)) a = ggml_cont(ctx, a);",
            "        a = ggml_repeat(ctx, a, b);",
            "        return {a, b};",
            "    }",
            "    int64_t target_ne[4];",
            "    bool need_repeat_a = false;",
            "    bool need_repeat_b = false;",
            "    for (int d = 0; d < 4; ++d) {",
            "        target_ne[d] = std::max(a->ne[d], b->ne[d]);",
            "        if (a->ne[d] != target_ne[d]) need_repeat_a = true;",
            "        if (b->ne[d] != target_ne[d]) need_repeat_b = true;",
            "    }",
            "    if (need_repeat_a) {",
            "        if (!ggml_is_contiguous(a)) a = ggml_cont(ctx, a);",
            "        a = ggml_repeat_4d(ctx, a, target_ne[0], target_ne[1], target_ne[2], target_ne[3]);",
            "    }",
            "    if (need_repeat_b) {",
            "        if (!ggml_is_contiguous(b)) b = ggml_cont(ctx, b);",
            "        b = ggml_repeat_4d(ctx, b, target_ne[0], target_ne[1], target_ne[2], target_ne[3]);",
            "    }",
            "    if (a->type != b->type) {",
            "        if (a->type == GGML_TYPE_I32 && b->type == GGML_TYPE_F32) {",
            "            if (!ggml_is_contiguous(a)) a = ggml_cont(ctx, a);",
            "            a = ggml_cast(ctx, a, GGML_TYPE_F32);",
            "        } else if (b->type == GGML_TYPE_I32 && a->type == GGML_TYPE_F32) {",
            "            if (!ggml_is_contiguous(b)) b = ggml_cont(ctx, b);",
            "            b = ggml_cast(ctx, b, GGML_TYPE_F32);",
            "        }",
            "    }",
            "    return {a, b};",
            "}",
            "",
            "// ----------------------------------------------------------------------------",
            "// Pairwise concat helper.",
            "// Skips empty inputs and aligns contiguity before concatenating.",
            "// ----------------------------------------------------------------------------",
            "inline struct ggml_tensor* concat_pair(",
            "    struct ggml_context* ctx,",
            "    struct ggml_tensor* a,",
            "    struct ggml_tensor* b,",
            "    int ggml_dim) {",
            "    auto is_empty = [](struct ggml_tensor* t) {",
            "        return t->ne[0] == 0 || t->ne[1] == 0 || t->ne[2] == 0 || t->ne[3] == 0;",
            "    };",
            "    if (is_empty(a)) return b;",
            "    if (is_empty(b)) return a;",
            "    if (!ggml_is_contiguous(a)) a = ggml_cont(ctx, a);",
            "    if (!ggml_is_contiguous(b)) b = ggml_cont(ctx, b);",
            "    return ggml_concat(ctx, a, b, ggml_dim);",
            "}",
            "",
            "// ----------------------------------------------------------------------------",
            "// Depthwise-conv layout helper.",
            "// _direct wants weight [KW,KH,1,C] x input [W,H,C,N]; fold 1D",
            "// layouts and cast to F32 (CUDA dw is F32-only). Mirrors the",
            "// interpreter.",
            "// ----------------------------------------------------------------------------",
            "inline std::pair<struct ggml_tensor*, struct ggml_tensor*> match_dw_layout(",
            "    struct ggml_context* ctx,",
            "    struct ggml_tensor* w,",
            "    struct ggml_tensor* x,",
            "    int is_1d) {",
            "    if (is_1d && w->ne[3] == 1) {",
            "        if (!ggml_is_contiguous(w)) w = ggml_cont(ctx, w);",
            "        w = ggml_cont(ctx, ggml_reshape_4d(ctx, w, w->ne[0], 1, w->ne[1], w->ne[2]));",
            "    }",
            "    if (is_1d && x->ne[3] == 1) {",
            "        if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);",
            "        x = ggml_cont(ctx, ggml_reshape_4d(ctx, x, x->ne[0], 1, x->ne[1], x->ne[2]));",
            "    }",
            "    if (w->ne[2] != 1 && w->ne[3] == 1) {",
            "        if (!ggml_is_contiguous(w)) w = ggml_cont(ctx, w);",
            "        w = ggml_cont(ctx, ggml_reshape_4d(ctx, w, w->ne[0], w->ne[1], 1, w->ne[2]));",
            "    }",
            "    if (w->type != GGML_TYPE_F32) w = ggml_cast(ctx, w, GGML_TYPE_F32);",
            "    if (x->type != GGML_TYPE_F32) x = ggml_cast(ctx, x, GGML_TYPE_F32);",
            "    return {w, x};",
            "}",
            "",
            "// ----------------------------------------------------------------------------",
            "// Reshape 4D contiguity helper.",
            "// ----------------------------------------------------------------------------",
            "inline struct ggml_tensor* reshape4d_contig(",
            "    struct ggml_context* ctx,",
            "    struct ggml_tensor* t,",
            "    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {",
            "    if (!t) return nullptr;",
            "    if (!ggml_is_contiguous(t)) t = ggml_cont(ctx, t);",
            "    return ggml_reshape_4d(ctx, t, ne0, ne1, ne2, ne3);",
            "}",
            "",
            "// ----------------------------------------------------------------------------",
            "// Convolution bias matching helper.",
            "// ----------------------------------------------------------------------------",
            "inline struct ggml_tensor* match_conv_bias(",
            "    struct ggml_context* ctx,",
            "    struct ggml_tensor* result,",
            "    struct ggml_tensor* bias) {",
            "    if (!bias || !result) return result;",
            "    if (!ggml_is_contiguous(bias)) bias = ggml_cont(ctx, bias);",
            "    if (bias->ne[0] == result->ne[2] && bias->ne[1] == 1 && bias->ne[2] == 1) {",
            "        bias = reshape4d_contig(ctx, bias, 1, 1, result->ne[2], 1);",
            "    }",
            "    if (!ggml_are_same_shape(bias, result) && ggml_can_repeat(bias, result)) {",
            "        bias = ggml_repeat(ctx, bias, result);",
            "    }",
            "    return ggml_add(ctx, result, bias);",
            "}",
            "",
            "// ----------------------------------------------------------------------------",
            "// Model Parameter Weights Structure",
            "// ----------------------------------------------------------------------------",
            "struct Weights {",
        ]

        # Declare weight pointers
        for pid in self.graph.parameters:
            t = self.graph.tensors[pid]
            ident = _sanitize_ident(t.name)
            lines.append(
                f"    struct ggml_tensor* {ident} = nullptr; // Tensor ID {t.id} ({t.ggml_type.name})"
            )

        lines.extend(
            [
                "",
                "    // Initialize weight tensor descriptors in the GGML context",
                "    void init_tensors(struct ggml_context* ctx, struct gguf_context* gguf_ctx) {",
            ]
        )

        for pid in self.graph.parameters:
            t = self.graph.tensors[pid]
            ident = _sanitize_ident(t.name)
            shape_str = ", ".join(_dim_to_cpp_expr(d) for d in t.ne)
            lines.append(f'        // Parameter: "{t.name}"')
            lines.append(f'        int64_t tid_{t.id} = gguf_find_tensor(gguf_ctx, "{t.name}");')
            lines.append(f"        if (tid_{t.id} >= 0) {{")
            lines.append(
                f"            this->{ident} = ggml_new_tensor_4d(ctx, static_cast<enum ggml_type>({int(t.ggml_type)}), {shape_str});"
            )
            lines.append(f'            ggml_set_name(this->{ident}, "{t.name}");')
            lines.append("        }")

        lines.extend(
            [
                "    }",
                "",
                "    // Load tensor data bytes from GGUF file into backend memory buffer",
                "    bool load_data(struct gguf_context* gguf_ctx, const std::string& filepath) {",
                "        std::ifstream fin(filepath, std::ios::binary);",
                "        if (!fin.is_open()) return false;",
                "        size_t data_offset = gguf_get_data_offset(gguf_ctx);",
            ]
        )

        for pid in self.graph.parameters:
            t = self.graph.tensors[pid]
            ident = _sanitize_ident(t.name)
            lines.extend(
                [
                    f'        int64_t tid_{t.id} = gguf_find_tensor(gguf_ctx, "{t.name}");',
                    f"        if (tid_{t.id} >= 0 && this->{ident}) {{",
                    f"            size_t t_offset = data_offset + gguf_get_tensor_offset(gguf_ctx, tid_{t.id});",
                    f"            size_t t_size = gguf_get_tensor_size(gguf_ctx, tid_{t.id});",
                    "            std::vector<uint8_t> buf(t_size);",
                    "            fin.seekg(static_cast<std::streamoff>(t_offset), std::ios::beg);",
                    "            fin.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(t_size));",
                    f"            ggml_backend_tensor_set(this->{ident}, buf.data(), 0, t_size);",
                    "        }",
                ]
            )

        lines.extend(
            [
                "        return true;",
                "    }",
                "",
                "    // Convenience unified load method",
                '    void load(struct ggml_context* ctx, struct gguf_context* gguf_ctx, const std::string& filepath = "") {',
                "        init_tensors(ctx, gguf_ctx);",
                "        if (!filepath.empty()) {",
                "            load_data(gguf_ctx, filepath);",
                "        }",
                "    }",
                "};",
                "",
                "// ----------------------------------------------------------------------------",
                "// Computation Graph Builder (build_graph)",
                "// Constructs the forward pass computational Directed Acyclic Graph (DAG).",
                "// ----------------------------------------------------------------------------",
                "inline struct ggml_cgraph* build_graph(",
                "    struct ggml_context* ctx,",
                "    const Weights& weights,",
                "    const std::unordered_map<std::string, struct ggml_tensor*>& inputs,",
                "    const std::unordered_map<std::string, int64_t>& symbols = {}",
                ") {",
                # Each graph op expands to several ggml kernels, so size the
                # arena generously from the node count instead of ggml's default.
                f"    size_t graph_nodes = std::max<size_t>(32768, {len(self.graph.nodes) * 16});",
                "    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, graph_nodes, false);",
                "",
                "    // Mapping from tensor ID to allocated computation node",
                "    std::unordered_map<uint32_t, struct ggml_tensor*> tensors;",
                "",
                "    // ========================================================================",
                "    // Phase 1: Bind Persistent Model Weights from Weights Struct",
                "    // ========================================================================",
            ]
        )

        for pid in self.graph.parameters:
            t = self.graph.tensors[pid]
            ident = _sanitize_ident(t.name)
            lines.append(f"    tensors[{t.id}] = weights.{ident};")

        lines.append("")
        lines.append(
            "    // ========================================================================"
        )
        lines.append("    // Phase 2: Ingest and Validate Named Model Inputs")
        lines.append(
            "    // ========================================================================"
        )
        for in_id in self.graph.inputs:
            t = self.graph.tensors[in_id]
            lines.append(f'    auto in_it_{in_id} = inputs.find("{t.name}");')
            lines.append(f"    if (in_it_{in_id} != inputs.end()) {{")
            lines.append(f"        tensors[{in_id}] = in_it_{in_id}->second;")
            lines.append("    }")

        lines.append("")
        lines.append(
            "    // ========================================================================"
        )
        lines.append("    // Phase 3: Forward Execution Graph Nodes (Operations & Activations)")
        lines.append(
            "    // ========================================================================"
        )

        # Emit computational nodes
        for node in self.graph.nodes:
            node_code = self._generate_node_code(node)
            lines.extend(node_code)

        lines.append("")
        lines.append(
            "    // ========================================================================"
        )
        lines.append("    // Phase 4: Register Model Outputs with Graph Forward Expander")
        lines.append(
            "    // ========================================================================"
        )
        for out_id in self.graph.outputs:
            lines.append(f"    if (tensors.find({out_id}) != tensors.end()) {{")
            lines.append(f"        ggml_build_forward_expand(gf, tensors[{out_id}]);")
            lines.append("    }")

        lines.extend(
            [
                "",
                "    return gf;",
                "}",
                "",
                f"}} // namespace {self.model_name}",
                "",
            ]
        )

        return "\n".join(lines)

    def _get_node_semantics_and_note(self, node: GGMLOpDef) -> tuple[str, str]:
        """Returns (high_level_semantics, lowering_notes) for a GGML op."""
        opc = node.opcode
        attrs = node.attributes

        if opc == GGMLOpCode.GGML_OP_MUL_MAT:
            if len(node.inputs) > 2:
                return (
                    "Linear Projection / GEMM + Bias: y = x @ W^T + b",
                    "GGML mul_mat consumes transposed weight as 1st argument (W, x); bias added after GEMM",
                )
            return (
                "Matrix Multiplication (GEMM): y = x @ W^T",
                "GGML mul_mat consumes transposed weight as 1st argument (W, x)",
            )
        if opc == GGMLOpCode.GGML_OP_ADD:
            return (
                "Elementwise Addition: y = a + b",
                "match_broadcast handles PyTorch-to-GGML dimension broadcasting",
            )
        if opc == GGMLOpCode.GGML_OP_SUB:
            return (
                "Elementwise Subtraction: y = a - b",
                "match_broadcast handles PyTorch-to-GGML dimension broadcasting",
            )
        if opc == GGMLOpCode.GGML_OP_MUL:
            return (
                "Elementwise Multiplication: y = a * b",
                "match_broadcast handles PyTorch-to-GGML dimension broadcasting",
            )
        if opc == GGMLOpCode.GGML_OP_DIV:
            return (
                "Elementwise Division: y = a / b",
                "match_broadcast handles PyTorch-to-GGML dimension broadcasting",
            )
        if opc == GGMLOpCode.GGML_OP_CUSTOM_SWIGLU:
            return (
                "Fused SwiGLU Activation: y = silu(gate) * up",
                "Fused kernel computes SwiGLU non-linearity in a single high-efficiency memory pass",
            )
        if opc == GGMLOpCode.GGML_OP_CUSTOM_RMS_NORM:
            eps = attrs.get("eps", 1e-5)
            return (
                f"Fused RMS Normalization: y = (x / rms(x, eps={eps})) * weight",
                "Standard LLM pre/post-attention root-mean-square normalization",
            )
        if opc == GGMLOpCode.GGML_OP_CUSTOM_LAYER_NORM:
            eps = attrs.get("eps", 1e-5)
            return (
                f"Fused Layer Normalization: y = ((x - mean) / sqrt(var + {eps})) * gamma + beta",
                "Standard LayerNorm with affine transformation",
            )
        if opc == GGMLOpCode.GGML_OP_CUSTOM_BIAS_GELU:
            return (
                "Fused Bias + GELU Activation: y = gelu(x + bias)",
                "Fused kernel computes bias addition and GELU non-linearity in single pass",
            )
        if opc == GGMLOpCode.GGML_OP_ROPE:
            n_dims = attrs.get("n_dims", "head_dim")
            mode = attrs.get("mode", 0)
            return (
                f"Rotary Position Embedding (RoPE): n_dims={n_dims}, mode={mode}",
                "Applies frequency rotations to query/key projections according to token position",
            )
        if opc == GGMLOpCode.GGML_OP_FLASH_ATTN_EXT:
            return (
                "Scaled Dot-Product Attention (FlashAttention): softmax(Q @ K^T / sqrt(d) + mask) @ V",
                "Fused multi-head attention kernel with causal masking and online softmax",
            )
        if opc == GGMLOpCode.GGML_OP_CONV_2D:
            s0, s1 = attrs.get("s0", 1), attrs.get("s1", 1)
            p0, p1 = attrs.get("p0", 0), attrs.get("p1", 0)
            return (
                f"2D Convolution: y = conv2d(x, W, stride=[{s0}, {s1}], pad=[{p0}, {p1}]) + bias",
                "PyTorch NCHW tensor converted to GGML WHCN memory layout for 2D convolution",
            )
        if opc == GGMLOpCode.GGML_OP_CONV_2D_DW:
            s0, s1 = attrs.get("s0", 1), attrs.get("s1", 1)
            p0, p1 = attrs.get("p0", 0), attrs.get("p1", 0)
            return (
                f"Depthwise 2D Convolution: stride=[{s0}, {s1}], pad=[{p0}, {p1}]",
                "Dedicated direct depthwise 2D convolution kernel",
            )
        if opc == GGMLOpCode.GGML_OP_POOL_2D:
            return ("2D Pooling (Max or Average)", "Spatial downsampling over HxW grid")
        if opc == GGMLOpCode.GGML_OP_GET_ROWS:
            return (
                "Embedding Lookup: y = embedding_table[token_indices]",
                "Token indices flattened to 1D vector before ggml_get_rows; result reshaped back to 4D",
            )
        if opc == GGMLOpCode.GGML_OP_UNARY:
            u_type = attrs.get("unary_op", "gelu")
            u_str = str(u_type).lower()
            if "erf" in u_str or u_type == 16:
                return (
                    "Exact GELU Activation: y = 0.5 * x * (1 + erf(x / sqrt(2)))",
                    "Mapped from PyTorch torch.nn.functional.gelu(approximate='none')",
                )
            if "gelu" in u_str or u_type == 8:
                return (
                    "Tanh-Approximated GELU Activation: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))",
                    "Mapped from PyTorch torch.nn.functional.gelu(approximate='tanh')",
                )
            if "silu" in u_str or u_type == 10:
                return ("SiLU / Swish Activation: y = x * sigmoid(x)", "GGML unary SiLU kernel")
            if "relu" in u_str or u_type == 6:
                return ("ReLU Activation: y = max(0, x)", "GGML unary ReLU kernel")
            if "sigmoid" in u_str or u_type == 7:
                return ("Sigmoid Activation: y = 1 / (1 + exp(-x))", "GGML unary Sigmoid kernel")
            if "hardswish" in u_str or u_type == 11:
                return (
                    "Hardswish Activation: y = x * relu6(x + 3) / 6",
                    "GGML unary Hardswish kernel",
                )
            if "hardsigmoid" in u_str or u_type == 12:
                return (
                    "Hardsigmoid Activation: y = relu6(x + 3) / 6",
                    "GGML unary Hardsigmoid kernel",
                )
            return (f"Unary Activation: {u_type}", "GGML unary kernel")
        if opc == GGMLOpCode.GGML_OP_REPEAT:
            return (
                "Broadcast / Repeat: expand tensor to match target dimensions",
                "Uses ggml_repeat_4d or reshape_4d",
            )
        if opc == GGMLOpCode.GGML_OP_VIEW:
            g_dim = attrs.get("ggml_dim", 0)
            start = attrs.get("start", 0)
            step = attrs.get("step", 1)
            return (
                f"Strided Sub-Tensor View / Slice: axis={g_dim}, start={start}, step={step}",
                "Zero-copy non-contiguous sub-tensor window into existing allocation",
            )
        if opc == GGMLOpCode.GGML_OP_RESHAPE:
            return (
                "In-Place Shape Reinterpretation",
                "Adjusts tensor dimension metadata without data copies",
            )
        if opc == GGMLOpCode.GGML_OP_PERMUTE:
            axes = [attrs.get(f"axis{i}", i) for i in range(4)]
            return (
                f"Multi-Axis Dimension Permutation: axes={axes}",
                "Permutes tensor strides without copying; ggml_cont materializes contiguous layout if required",
            )
        if opc == GGMLOpCode.GGML_OP_TRANSPOSE:
            return ("2D Matrix Transposition: swap rows and columns", "Reorders row/column strides")
        if opc == GGMLOpCode.GGML_OP_SOFT_MAX:
            return (
                "Softmax Normalization: y = exp(x - max(x)) / sum(exp(x - max(x)))",
                "Computes row-wise softmax",
            )
        if opc == GGMLOpCode.GGML_OP_CLAMP:
            min_v = attrs.get("min", "-inf")
            max_v = attrs.get("max", "+inf")
            return (
                f"Value Clamping: y = clamp(x, min={min_v}, max={max_v})",
                "Restricts tensor elements within range",
            )
        if opc == GGMLOpCode.GGML_OP_CONCAT:
            dim = attrs.get("dim", 0)
            return (
                f"Tensor Concatenation: concat along axis {dim}",
                "Assembles multiple input tensors into one",
            )
        if opc == GGMLOpCode.GGML_OP_NORM:
            return (
                "L2 Normalization / Euclidean Norm along rows",
                "Computes Euclidean norm of input vectors",
            )
        if opc == GGMLOpCode.GGML_OP_MEAN:
            g_dim = attrs.get("ggml_dim", 0)
            return (
                f"Mean Reduction along axis {g_dim}",
                f"GGML only reduces rows (axis 0); axis {g_dim} permuted to axis 0 before mean",
            )
        if opc in (GGMLOpCode.GGML_OP_SUM, GGMLOpCode.GGML_OP_SUM_ROWS):
            g_dim = attrs.get("ggml_dim", 0)
            return (
                f"Sum Reduction along axis {g_dim}",
                f"GGML only reduces rows (axis 0); axis {g_dim} permuted to axis 0 before sum",
            )
        if opc == GGMLOpCode.GGML_OP_ARGMAX:
            return (
                "Argmax Operation: y = argmax(x, dim=0)",
                "Finds index of maximum value along row dimension",
            )
        if opc == GGMLOpCode.GGML_OP_CONT:
            return (
                "Contiguity Enforcement",
                "Allocates contiguous buffer to resolve non-contiguous memory layouts",
            )
        if opc == GGMLOpCode.GGML_OP_PAD:
            p0, p1 = attrs.get("p0", 0), attrs.get("p1", 0)
            return (f"Zero Padding: pad=[{p0}, {p1}]", "Pads spatial dimensions with zeros")
        if opc == GGMLOpCode.GGML_OP_SQR:
            exp = attrs.get("exponent", 2)
            return (f"Power / Square Operation: y = x^{exp}", "Computes elementwise power")
        if opc == GGMLOpCode.GGML_OP_SIN:
            return ("Elementwise Sine: y = sin(x)", "GGML trigonometric sine kernel")
        if opc == GGMLOpCode.GGML_OP_COS:
            return ("Elementwise Cosine: y = cos(x)", "GGML trigonometric cosine kernel")
        if opc == GGMLOpCode.GGML_OP_LOG:
            return ("Elementwise Natural Logarithm: y = ln(x)", "GGML natural logarithm kernel")
        if opc == GGMLOpCode.GGML_OP_GLU:
            return ("Gated Linear Unit (GLU): y = a * sigmoid(b)", "GGML GLU kernel")

        return (f"Operation: {node.opcode.name}", "")

    def _format_node_header_comment(self, node: GGMLOpDef) -> list[str]:
        """Generates rich, human-readable commentary documenting PyTorch source, semantics, and lowering."""
        lines = []
        op_name = node.name or f"node_{node.id}"
        lines.append(
            "    // ------------------------------------------------------------------------"
        )
        lines.append(f"    // Node {node.id}: {op_name} [{node.opcode.name}]")

        semantics, note = self._get_node_semantics_and_note(node)
        if semantics:
            lines.append(f"    // Semantics: {semantics}")

        src_op = node.attributes.get("source_op")
        mod_path = node.attributes.get("module_path")
        fused_pass = node.attributes.get("fused_pass")
        if src_op:
            lines.append(f"    // Source Op: {src_op}")
        if mod_path:
            lines.append(f"    // PyTorch Module: {mod_path}")
        if fused_pass:
            lines.append(f"    // Optimization Pass: {fused_pass}")

        lines.append("    // Operands:")
        for idx, in_id in enumerate(node.inputs):
            in_t = self.graph.tensors.get(in_id)
            if in_t:
                role = (
                    "Parameter"
                    if in_id in self.graph.parameters
                    else ("Model Input" if in_id in self.graph.inputs else "Activation")
                )
                shape_str = "x".join(_dim_to_cpp_expr(d) for d in in_t.ne)
                lines.append(
                    f'    //   in[{idx}]: tensors[{in_id}] ("{in_t.name}", {role}, [{shape_str}], {in_t.ggml_type.name})'
                )
            else:
                lines.append(f"    //   in[{idx}]: tensors[{in_id}]")

        for idx, out_id in enumerate(node.outputs):
            out_t = self.graph.tensors.get(out_id)
            if out_t:
                role = "Model Output" if out_id in self.graph.outputs else "Activation"
                shape_str = "x".join(_dim_to_cpp_expr(d) for d in out_t.ne)
                lines.append(
                    f'    //   out[{idx}]: tensors[{out_id}] ("{out_t.name}", {role}, [{shape_str}], {out_t.ggml_type.name})'
                )
            else:
                lines.append(f"    //   out[{idx}]: tensors[{out_id}]")

        if note:
            lines.append(f"    // Lowering: {note}")

        lines.append(
            "    // ------------------------------------------------------------------------"
        )
        return lines

    def _generate_node_code(self, node: GGMLOpDef) -> list[str]:
        lines = self._format_node_header_comment(node)

        out_id = node.outputs[0]
        inp_vars = [f"tensors[{i}]" for i in node.inputs]

        if node.opcode == GGMLOpCode.GGML_OP_ADD:
            lines.append(
                f"    auto bc_{node.id} = match_broadcast(ctx, {inp_vars[0]}, {inp_vars[1]});"
            )
            lines.append(
                f"    tensors[{out_id}] = ggml_add(ctx, bc_{node.id}.first, bc_{node.id}.second);"
            )
        elif node.opcode == GGMLOpCode.GGML_OP_SUB:
            lines.append(
                f"    auto bc_{node.id} = match_broadcast(ctx, {inp_vars[0]}, {inp_vars[1]});"
            )
            lines.append(
                f"    tensors[{out_id}] = ggml_sub(ctx, bc_{node.id}.first, bc_{node.id}.second);"
            )
        elif node.opcode == GGMLOpCode.GGML_OP_MUL:
            lines.append(
                f"    auto bc_{node.id} = match_broadcast(ctx, {inp_vars[0]}, {inp_vars[1]});"
            )
            lines.append(
                f"    tensors[{out_id}] = ggml_mul(ctx, bc_{node.id}.first, bc_{node.id}.second);"
            )
        elif node.opcode == GGMLOpCode.GGML_OP_DIV:
            lines.append(
                f"    auto bc_{node.id} = match_broadcast(ctx, {inp_vars[0]}, {inp_vars[1]});"
            )
            lines.append(
                f"    tensors[{out_id}] = ggml_div(ctx, bc_{node.id}.first, bc_{node.id}.second);"
            )
        elif node.opcode == GGMLOpCode.GGML_OP_MUL_MAT:
            if node.attributes.get("transpose_in0", 0) == 1:
                # transpose does not need its source to be contiguous, but the
                # result of the transpose must be made contiguous for the matmul
                lines.append(
                    f"    struct ggml_tensor* mm0_{node.id} = ggml_cont(ctx, ggml_transpose(ctx, {inp_vars[0]}));"
                )
            else:
                lines.append(f"    struct ggml_tensor* mm0_{node.id} = {inp_vars[0]};")
                lines.append(
                    f"    if (!ggml_is_contiguous(mm0_{node.id})) mm0_{node.id} = ggml_cont(ctx, mm0_{node.id});"
                )
            lines.append(f"    struct ggml_tensor* mm1_{node.id} = {inp_vars[1]};")
            lines.append(
                f"    if (!ggml_is_contiguous(mm1_{node.id})) mm1_{node.id} = ggml_cont(ctx, mm1_{node.id});"
            )
            lines.append(
                f"    tensors[{out_id}] = ggml_mul_mat(ctx, mm0_{node.id}, mm1_{node.id});"
            )
            b_arg = inp_vars[2] if len(inp_vars) > 2 else "nullptr"
            if b_arg != "nullptr":
                lines.append(f"    struct ggml_tensor* b_{node.id} = {b_arg};")
                lines.append(
                    f"    if (!ggml_is_contiguous(b_{node.id})) b_{node.id} = ggml_cont(ctx, b_{node.id});"
                )
                lines.append(
                    f"    tensors[{out_id}] = ggml_add(ctx, tensors[{out_id}], b_{node.id});"
                )
        elif node.opcode == GGMLOpCode.GGML_OP_UNARY:
            u_type = node.attributes.get("unary_op", "gelu")
            u_str = str(u_type).lower()
            if "erf" in u_str or u_type == 16:
                lines.append(
                    f"    tensors[{out_id}] = ggml_unary(ctx, {inp_vars[0]}, GGML_UNARY_OP_GELU_ERF);"
                )
            elif "gelu" in u_str or u_type == 8:
                lines.append(
                    f"    tensors[{out_id}] = ggml_unary(ctx, {inp_vars[0]}, GGML_UNARY_OP_GELU);"
                )
            elif "silu" in u_str or u_type == 10:
                lines.append(
                    f"    tensors[{out_id}] = ggml_unary(ctx, {inp_vars[0]}, GGML_UNARY_OP_SILU);"
                )
            elif "relu" in u_str or u_type == 6:
                lines.append(
                    f"    tensors[{out_id}] = ggml_unary(ctx, {inp_vars[0]}, GGML_UNARY_OP_RELU);"
                )
            elif "sigmoid" in u_str or u_type == 7:
                lines.append(
                    f"    tensors[{out_id}] = ggml_unary(ctx, {inp_vars[0]}, GGML_UNARY_OP_SIGMOID);"
                )
            elif "hardswish" in u_str or u_type == 11:
                lines.append(
                    f"    tensors[{out_id}] = ggml_unary(ctx, {inp_vars[0]}, GGML_UNARY_OP_HARDSWISH);"
                )
            elif "hardsigmoid" in u_str or u_type == 12:
                lines.append(
                    f"    tensors[{out_id}] = ggml_unary(ctx, {inp_vars[0]}, GGML_UNARY_OP_HARDSIGMOID);"
                )
            elif "tanh" in u_str or u_type == 4:
                lines.append(
                    f"    tensors[{out_id}] = ggml_unary(ctx, {inp_vars[0]}, GGML_UNARY_OP_TANH);"
                )
            elif "exp" in u_str or u_type == 13:
                lines.append(
                    f"    tensors[{out_id}] = ggml_unary(ctx, {inp_vars[0]}, GGML_UNARY_OP_EXP);"
                )
            elif "neg" in u_str or u_type == 2:
                lines.append(
                    f"    tensors[{out_id}] = ggml_unary(ctx, {inp_vars[0]}, GGML_UNARY_OP_NEG);"
                )
            elif "abs" in u_str or u_type == 0:
                lines.append(
                    f"    tensors[{out_id}] = ggml_unary(ctx, {inp_vars[0]}, GGML_UNARY_OP_ABS);"
                )
            else:
                lines.append(
                    f"    tensors[{out_id}] = ggml_unary(ctx, {inp_vars[0]}, GGML_UNARY_OP_RELU);"
                )
        elif node.opcode == GGMLOpCode.GGML_OP_CLAMP:
            if "min" in node.attributes:
                min_expr = f"{node.attributes['min']}f"
            else:
                min_expr = "-FLT_MAX"
            if "max" in node.attributes:
                max_expr = f"{node.attributes['max']}f"
            else:
                max_expr = "FLT_MAX"
            lines.append(
                f"    tensors[{out_id}] = ggml_clamp(ctx, {inp_vars[0]}, {min_expr}, {max_expr});"
            )
        elif node.opcode == GGMLOpCode.GGML_OP_CONV_2D:
            s0 = node.attributes.get("stride_w", 1)
            s1 = node.attributes.get("stride_h", 1)
            p0 = node.attributes.get("pad_w", 0)
            p1 = node.attributes.get("pad_h", 0)
            d0 = node.attributes.get("dilation_w", 1)
            d1 = node.attributes.get("dilation_h", 1)
            is_1d = int(node.attributes.get("is_1d", 0))
            lines.append(f"    struct ggml_tensor* c2_w_{node.id} = {inp_vars[0]};")
            lines.append(f"    struct ggml_tensor* c2_x_{node.id} = {inp_vars[1]};")
            if is_1d:
                lines.append(
                    f"    if (c2_w_{node.id}->ne[3] == 1) c2_w_{node.id} = reshape4d_contig(ctx, c2_w_{node.id}, c2_w_{node.id}->ne[0], 1, c2_w_{node.id}->ne[1], c2_w_{node.id}->ne[2]);"
                )
                lines.append(
                    f"    if (c2_x_{node.id}->ne[3] == 1) c2_x_{node.id} = reshape4d_contig(ctx, c2_x_{node.id}, c2_x_{node.id}->ne[0], 1, c2_x_{node.id}->ne[1], c2_x_{node.id}->ne[2]);"
                )
            lines.append(
                f"    tensors[{out_id}] = ggml_conv_2d(ctx, c2_w_{node.id}, c2_x_{node.id}, {s0}, {s1}, {p0}, {p1}, {d0}, {d1});"
            )
            if len(inp_vars) > 2:
                lines.append(
                    f"    tensors[{out_id}] = match_conv_bias(ctx, tensors[{out_id}], {inp_vars[2]});"
                )
            if node.attributes.get("fused_relu", 0):
                lines.append(f"    tensors[{out_id}] = ggml_relu(ctx, tensors[{out_id}]);")
            if is_1d:
                lines.append(
                    f"    if (tensors[{out_id}] && tensors[{out_id}]->ne[1] == 1) tensors[{out_id}] = reshape4d_contig(ctx, tensors[{out_id}], tensors[{out_id}]->ne[0], tensors[{out_id}]->ne[2], tensors[{out_id}]->ne[3], 1);"
                )
        elif node.opcode == GGMLOpCode.GGML_OP_CONV_2D_DW:
            s0 = node.attributes.get("stride_w", 1)
            s1 = node.attributes.get("stride_h", 1)
            p0 = node.attributes.get("pad_w", 0)
            p1 = node.attributes.get("pad_h", 0)
            d0 = node.attributes.get("dilation_w", 1)
            d1 = node.attributes.get("dilation_h", 1)
            is_1d = int(node.attributes.get("is_1d", 0))
            lines.append(
                f"    auto dw_{node.id} = match_dw_layout(ctx, {inp_vars[0]}, {inp_vars[1]}, {is_1d});"
            )
            lines.append(
                f"    tensors[{out_id}] = ggml_conv_2d_dw_direct(ctx, dw_{node.id}.first, dw_{node.id}.second, {s0}, {s1}, {p0}, {p1}, {d0}, {d1});"
            )
            if len(inp_vars) > 2:
                lines.append(
                    f"    tensors[{out_id}] = match_conv_bias(ctx, tensors[{out_id}], {inp_vars[2]});"
                )
            if node.attributes.get("fused_relu", 0):
                lines.append(f"    tensors[{out_id}] = ggml_relu(ctx, tensors[{out_id}]);")
            if is_1d:
                lines.append(
                    f"    if (tensors[{out_id}] && tensors[{out_id}]->ne[1] == 1) tensors[{out_id}] = reshape4d_contig(ctx, tensors[{out_id}], tensors[{out_id}]->ne[0], tensors[{out_id}]->ne[2], tensors[{out_id}]->ne[3], 1);"
                )
        elif node.opcode == GGMLOpCode.GGML_OP_POOL_2D:
            is_max = node.attributes.get("is_max", 0) != 0
            pool_enum = "GGML_OP_POOL_MAX" if is_max else "GGML_OP_POOL_AVG"
            is_adapt = node.attributes.get("is_adaptive", 0) != 0
            if is_adapt:
                lines.append(
                    f"    tensors[{out_id}] = ggml_pool_2d(ctx, {inp_vars[0]}, {pool_enum}, {inp_vars[0]}->ne[0], {inp_vars[0]}->ne[1], {inp_vars[0]}->ne[0], {inp_vars[0]}->ne[1], 0.0f, 0.0f);"
                )
            else:
                k0 = node.attributes.get("ksize_w", 2)
                k1 = node.attributes.get("ksize_h", 2)
                s0 = node.attributes.get("stride_w", k0)
                s1 = node.attributes.get("stride_h", k1)
                p0 = node.attributes.get("pad_w", 0)
                p1 = node.attributes.get("pad_h", 0)
                lines.append(
                    f"    tensors[{out_id}] = ggml_pool_2d(ctx, {inp_vars[0]}, {pool_enum}, {k0}, {k1}, {s0}, {s1}, {p0}.0f, {p1}.0f);"
                )
        elif node.opcode == GGMLOpCode.GGML_OP_SOFT_MAX:
            lines.append(f"    tensors[{out_id}] = ggml_soft_max(ctx, {inp_vars[0]});")
        elif node.opcode == GGMLOpCode.GGML_OP_REPEAT:
            if len(inp_vars) == 1:
                out_t = self.graph.tensors[out_id]
                ne_strs = [_dim_to_cpp_expr(d) for d in out_t.ne]
                lines.append(f"    struct ggml_tensor* rep_in_{node.id} = {inp_vars[0]};")
                lines.append(
                    f"    if (!ggml_is_contiguous(rep_in_{node.id})) rep_in_{node.id} = ggml_cont(ctx, rep_in_{node.id});"
                )
                lines.append(
                    f"    int64_t rep_in_el_{node.id} = rep_in_{node.id}->ne[0] * rep_in_{node.id}->ne[1] * rep_in_{node.id}->ne[2] * rep_in_{node.id}->ne[3];"
                )
                lines.append(
                    f"    int64_t rep_out_el_{node.id} = (int64_t)({ne_strs[0]}) * (int64_t)({ne_strs[1]}) * (int64_t)({ne_strs[2]}) * (int64_t)({ne_strs[3]});"
                )
                lines.append(f"    if (rep_in_el_{node.id} == rep_out_el_{node.id}) {{")
                lines.append(
                    f"        tensors[{out_id}] = ggml_reshape_4d(ctx, rep_in_{node.id}, {', '.join(ne_strs)});"
                )
                lines.append("    } else {")
                lines.append(
                    f"        tensors[{out_id}] = ggml_repeat_4d(ctx, rep_in_{node.id}, {', '.join(ne_strs)});"
                )
                lines.append("    }")
            else:
                lines.append(
                    f"    tensors[{out_id}] = ggml_repeat(ctx, {inp_vars[0]}, {inp_vars[1]});"
                )
        elif node.opcode == GGMLOpCode.GGML_OP_RESHAPE:
            out_t = self.graph.tensors[out_id]
            ne_strs = [_dim_to_cpp_expr(d) for d in out_t.ne]
            lines.append(f"    struct ggml_tensor* idx_{node.id} = {inp_vars[0]};")
            lines.append(
                f"    if (!ggml_is_contiguous(idx_{node.id})) idx_{node.id} = ggml_cont(ctx, idx_{node.id});"
            )
            lines.append(
                f"    tensors[{out_id}] = ggml_reshape_4d(ctx, idx_{node.id}, {', '.join(ne_strs)});"
            )
        elif node.opcode == GGMLOpCode.GGML_OP_PERMUTE:
            ax = [
                node.attributes.get("axis0", 0),
                node.attributes.get("axis1", 1),
                node.attributes.get("axis2", 2),
                node.attributes.get("axis3", 3),
            ]
            lines.append(
                f"    tensors[{out_id}] = ggml_permute(ctx, {inp_vars[0]}, {ax[0]}, {ax[1]}, {ax[2]}, {ax[3]});"
            )
        elif node.opcode == GGMLOpCode.GGML_OP_TRANSPOSE:
            lines.append(f"    tensors[{out_id}] = ggml_transpose(ctx, {inp_vars[0]});")
        elif node.opcode == GGMLOpCode.GGML_OP_NORM:
            w_arg = inp_vars[1] if len(inp_vars) > 1 else "nullptr"
            b_arg = inp_vars[2] if len(inp_vars) > 2 else "nullptr"
            eps = node.attributes.get("eps", 1e-5)
            lines.append(f"    tensors[{out_id}] = ggml_norm(ctx, {inp_vars[0]}, {eps}f);")
            if w_arg != "nullptr":
                lines.append(
                    f"    if ({w_arg}) {{ struct ggml_tensor* w = {w_arg}; if (ggml_can_repeat(w, tensors[{out_id}])) w = ggml_repeat(ctx, w, tensors[{out_id}]); tensors[{out_id}] = ggml_mul(ctx, tensors[{out_id}], w); }}"
                )
            if b_arg != "nullptr":
                lines.append(
                    f"    if ({b_arg}) {{ struct ggml_tensor* b = {b_arg}; if (ggml_can_repeat(b, tensors[{out_id}])) b = ggml_repeat(ctx, b, tensors[{out_id}]); tensors[{out_id}] = ggml_add(ctx, tensors[{out_id}], b); }}"
                )
        elif node.opcode == GGMLOpCode.GGML_OP_RMS_NORM:
            w_arg = inp_vars[1] if len(inp_vars) > 1 else "nullptr"
            eps = node.attributes.get("eps", 1e-5)
            lines.append(f"    tensors[{out_id}] = ggml_rms_norm(ctx, {inp_vars[0]}, {eps}f);")
            if w_arg != "nullptr":
                lines.append(
                    f"    if ({w_arg}) {{ struct ggml_tensor* w = {w_arg}; if (ggml_can_repeat(w, tensors[{out_id}])) w = ggml_repeat(ctx, w, tensors[{out_id}]); tensors[{out_id}] = ggml_mul(ctx, tensors[{out_id}], w); }}"
                )
        elif node.opcode == GGMLOpCode.GGML_OP_FLASH_ATTN_EXT:
            q_var = inp_vars[0]
            k_var = inp_vars[1]
            v_var = inp_vars[2] if len(inp_vars) > 2 else "nullptr"
            mask_var = inp_vars[3] if len(inp_vars) > 3 else "nullptr"
            # Absent scale means the PyTorch default 1/sqrt(head_dim),
            # computed from q at runtime since the dim may be symbolic.
            if "scale" in node.attributes:
                scale_expr = f"{node.attributes['scale']}f"
            else:
                scale_expr = f"(1.0f / sqrtf((float)q_{node.id}->ne[0]))"
            lines.append(
                f"    struct ggml_tensor* q_{node.id} = {q_var}; if (!ggml_is_contiguous(q_{node.id})) q_{node.id} = ggml_cont(ctx, q_{node.id});"
            )
            lines.append(
                f"    struct ggml_tensor* k_{node.id} = {k_var}; if (!ggml_is_contiguous(k_{node.id})) k_{node.id} = ggml_cont(ctx, k_{node.id});"
            )
            lines.append(
                f"    struct ggml_tensor* v_{node.id} = {v_var}; if (v_{node.id} && !ggml_is_contiguous(v_{node.id})) v_{node.id} = ggml_cont(ctx, v_{node.id});"
            )
            lines.append(
                f"    tensors[{out_id}] = ggml_flash_attn_ext(ctx, q_{node.id}, k_{node.id}, v_{node.id}, {mask_var}, {scale_expr}, 0.0f, 0.0f);"
            )
            # Raw output is heads-major; the graph layout needs L/H swapped
            # unless the transpose is already fused upstream.
            if not node.attributes.get("fused_transpose", 0):
                lines.append(
                    f"    tensors[{out_id}] = ggml_permute(ctx, tensors[{out_id}], 0, 2, 1, 3);"
                )
        elif node.opcode == GGMLOpCode.GGML_OP_ROPE:
            n_dims = node.attributes.get("n_dims", 0)
            mode = node.attributes.get("mode", 0)
            freq_base = node.attributes.get("freq_base", 10000.0)
            freq_scale = node.attributes.get("freq_scale", 1.0)
            n_dims_expr = f"{inp_vars[0]}->ne[0]" if n_dims == 0 else str(n_dims)
            if "freq_base" in node.attributes:
                lines.append(
                    f"    tensors[{out_id}] = ggml_rope_ext(ctx, {inp_vars[0]}, {inp_vars[1]}, nullptr, {n_dims_expr}, {mode}, 0, {freq_base}f, {freq_scale}f, 0.0f, 1.0f, 0.0f, 0.0f);"
                )
            else:
                lines.append(
                    f"    tensors[{out_id}] = ggml_rope(ctx, {inp_vars[0]}, {inp_vars[1]}, {n_dims_expr}, {mode});"
                )
        elif node.opcode == GGMLOpCode.GGML_OP_GET_ROWS:
            out_t = self.graph.tensors[out_id]
            ne_strs = [_dim_to_cpp_expr(d) for d in out_t.ne]
            lines.append(f"    struct ggml_tensor* idx_{node.id} = {inp_vars[1]};")
            lines.append(
                f"    if (idx_{node.id}->type != GGML_TYPE_I32) idx_{node.id} = ggml_cast(ctx, idx_{node.id}, GGML_TYPE_I32);"
            )
            lines.append(
                f"    if (!ggml_is_contiguous(idx_{node.id})) idx_{node.id} = ggml_cont(ctx, idx_{node.id});"
            )
            lines.append(
                f"    int64_t total_idx_{node.id} = idx_{node.id}->ne[0] * idx_{node.id}->ne[1] * idx_{node.id}->ne[2] * idx_{node.id}->ne[3];"
            )
            lines.append(
                f"    struct ggml_tensor* idx_flat_{node.id} = ggml_reshape_1d(ctx, idx_{node.id}, total_idx_{node.id});"
            )
            lines.append(
                f"    struct ggml_tensor* gr_{node.id} = ggml_get_rows(ctx, {inp_vars[0]}, idx_flat_{node.id});"
            )
            lines.append(
                f"    if (!ggml_is_contiguous(gr_{node.id})) gr_{node.id} = ggml_cont(ctx, gr_{node.id});"
            )
            lines.append(
                f"    tensors[{out_id}] = ggml_reshape_4d(ctx, gr_{node.id}, {', '.join(ne_strs)});"
            )
        elif node.opcode == GGMLOpCode.GGML_OP_CONCAT:
            dim = node.attributes.get("ggml_dim", node.attributes.get("dim", 0))
            if len(inp_vars) > 2:
                parts = ", ".join(inp_vars)
                lines.append(f"    struct ggml_tensor* cc_parts_{node.id}[] = {{{parts}}};")
                lines.append(f"    struct ggml_tensor* cc_{node.id} = cc_parts_{node.id}[0];")
                lines.append(
                    f"    for (int ci_{node.id} = 1; ci_{node.id} < {len(inp_vars)}; ++ci_{node.id}) {{"
                )
                lines.append(
                    f"        cc_{node.id} = concat_pair(ctx, cc_{node.id}, cc_parts_{node.id}[ci_{node.id}], {dim});"
                )
                lines.append("    }")
            else:
                lines.append(f"    struct ggml_tensor* cc_{node.id} = {inp_vars[0]};")
                if len(inp_vars) > 1:
                    lines.append(
                        f"    cc_{node.id} = concat_pair(ctx, cc_{node.id}, {inp_vars[1]}, {dim});"
                    )
            lines.append(
                f"    if (cc_{node.id} && !ggml_is_contiguous(cc_{node.id})) cc_{node.id} = ggml_cont(ctx, cc_{node.id});"
            )
            lines.append(f"    tensors[{out_id}] = cc_{node.id};")
        elif node.opcode == GGMLOpCode.GGML_OP_CONT:
            lines.append(f"    tensors[{out_id}] = ggml_cont(ctx, {inp_vars[0]});")
        elif node.opcode == GGMLOpCode.GGML_OP_SCALE:
            s_val = node.attributes.get("scale", 1.0)
            lines.append(f"    tensors[{out_id}] = ggml_scale(ctx, {inp_vars[0]}, {s_val}f);")
        elif node.opcode == GGMLOpCode.GGML_OP_SQRT:
            lines.append(f"    struct ggml_tensor* sq_{node.id} = {inp_vars[0]};")
            lines.append(
                f"    if (!ggml_is_contiguous(sq_{node.id})) sq_{node.id} = ggml_cont(ctx, sq_{node.id});"
            )
            if node.attributes.get("is_rsqrt", 0):
                lines.append(
                    f"    struct ggml_tensor* sqr_{node.id} = ggml_sqrt(ctx, sq_{node.id});"
                )
                lines.append(f"    tensors[{out_id}] = ggml_div(ctx, sqr_{node.id}, sq_{node.id});")
            else:
                lines.append(f"    tensors[{out_id}] = ggml_sqrt(ctx, sq_{node.id});")
        elif node.opcode == GGMLOpCode.GGML_OP_SUM_ROWS:
            g_dim = node.attributes.get("ggml_dim", 0)
            out_t = self.graph.tensors[out_id]
            ne_strs = [_dim_to_cpp_expr(d) for d in out_t.ne]
            if g_dim == 1:
                lines.append(
                    f"    struct ggml_tensor* sr_{node.id} = ggml_cont(ctx, ggml_transpose(ctx, {inp_vars[0]}));"
                )
                lines.append(f"    sr_{node.id} = ggml_sum_rows(ctx, sr_{node.id});")
                lines.append(
                    f"    sr_{node.id} = ggml_cont(ctx, ggml_transpose(ctx, sr_{node.id}));"
                )
            elif g_dim >= 2:
                # Single-axis reduction: swap axis g_dim to position 0,
                # reduce, swap back (ggml only reduces rows).
                swap = "2, 1, 0, 3" if g_dim == 2 else "3, 1, 2, 0"
                lines.append(f"    struct ggml_tensor* sr_{node.id} = {inp_vars[0]};")
                lines.append(
                    f"    if (!ggml_is_contiguous(sr_{node.id})) sr_{node.id} = ggml_cont(ctx, sr_{node.id});"
                )
                lines.append(
                    f"    sr_{node.id} = ggml_cont(ctx, ggml_permute(ctx, sr_{node.id}, {swap}));"
                )
                lines.append(f"    sr_{node.id} = ggml_sum_rows(ctx, sr_{node.id});")
                lines.append(
                    f"    sr_{node.id} = ggml_cont(ctx, ggml_permute(ctx, sr_{node.id}, {swap}));"
                )
            else:
                lines.append(f"    struct ggml_tensor* sr_{node.id} = {inp_vars[0]};")
                lines.append(
                    f"    if (!ggml_is_contiguous(sr_{node.id})) sr_{node.id} = ggml_cont(ctx, sr_{node.id});"
                )
                lines.append(f"    sr_{node.id} = ggml_sum_rows(ctx, sr_{node.id});")
            lines.append(
                f"    if (!ggml_is_contiguous(sr_{node.id})) sr_{node.id} = ggml_cont(ctx, sr_{node.id});"
            )
            lines.append(
                f"    tensors[{out_id}] = ggml_reshape_4d(ctx, sr_{node.id}, {', '.join(ne_strs)});"
            )
        elif node.opcode == GGMLOpCode.GGML_OP_MEAN:
            g_dim = node.attributes.get("ggml_dim", 0)
            out_t = self.graph.tensors[out_id]
            ne_strs = [_dim_to_cpp_expr(d) for d in out_t.ne]
            if g_dim == 1:
                lines.append(
                    f"    struct ggml_tensor* mn_{node.id} = ggml_cont(ctx, ggml_transpose(ctx, {inp_vars[0]}));"
                )
                lines.append(f"    mn_{node.id} = ggml_mean(ctx, mn_{node.id});")
                lines.append(
                    f"    mn_{node.id} = ggml_cont(ctx, ggml_transpose(ctx, mn_{node.id}));"
                )
            elif g_dim >= 2:
                # Single-axis reduction: swap axis g_dim to position 0,
                # reduce, swap back (ggml only reduces rows).
                swap = "2, 1, 0, 3" if g_dim == 2 else "3, 1, 2, 0"
                lines.append(f"    struct ggml_tensor* mn_{node.id} = {inp_vars[0]};")
                lines.append(
                    f"    if (!ggml_is_contiguous(mn_{node.id})) mn_{node.id} = ggml_cont(ctx, mn_{node.id});"
                )
                lines.append(
                    f"    mn_{node.id} = ggml_cont(ctx, ggml_permute(ctx, mn_{node.id}, {swap}));"
                )
                lines.append(f"    mn_{node.id} = ggml_mean(ctx, mn_{node.id});")
                lines.append(
                    f"    mn_{node.id} = ggml_cont(ctx, ggml_permute(ctx, mn_{node.id}, {swap}));"
                )
            else:
                lines.append(f"    struct ggml_tensor* mn_{node.id} = {inp_vars[0]};")
                lines.append(
                    f"    if (!ggml_is_contiguous(mn_{node.id})) mn_{node.id} = ggml_cont(ctx, mn_{node.id});"
                )
                lines.append(f"    mn_{node.id} = ggml_mean(ctx, mn_{node.id});")
            lines.append(
                f"    if (!ggml_is_contiguous(mn_{node.id})) mn_{node.id} = ggml_cont(ctx, mn_{node.id});"
            )
            lines.append(
                f"    tensors[{out_id}] = ggml_reshape_4d(ctx, mn_{node.id}, {', '.join(ne_strs)});"
            )
        elif node.opcode == GGMLOpCode.GGML_OP_CPY:
            out_t = self.graph.tensors[out_id]
            ne_strs = [_dim_to_cpp_expr(d) for d in out_t.ne]
            lines.append(
                f"    struct ggml_tensor* dst_{node.id} = ggml_new_tensor_4d(ctx, static_cast<enum ggml_type>({int(out_t.ggml_type)}), {', '.join(ne_strs)});"
            )
            lines.append(f"    tensors[{out_id}] = ggml_cpy(ctx, {inp_vars[0]}, dst_{node.id});")
        elif node.opcode == GGMLOpCode.GGML_OP_ARGMAX:
            lines.append(f"    tensors[{out_id}] = ggml_argmax(ctx, {inp_vars[0]});")
        elif node.opcode == GGMLOpCode.GGML_OP_DIAG_MASK_INF:
            n_past = node.attributes.get("n_past", 0)
            lines.append(
                f"    tensors[{out_id}] = ggml_diag_mask_inf(ctx, {inp_vars[0]}, {n_past});"
            )
        elif node.opcode == GGMLOpCode.GGML_OP_VIEW:
            out_t = self.graph.tensors[out_id]
            ne_strs = [_dim_to_cpp_expr(d) for d in out_t.ne]
            start = node.attributes.get("start", 0)
            ggml_dim = node.attributes.get("ggml_dim", 0)
            mult = node.attributes.get("offset_mult", 1)
            step = node.attributes.get("step", 1)
            if mult == 1:
                offset = f"{start} * {inp_vars[0]}->nb[{ggml_dim}]"
            else:
                offset = f"{start} * {mult} * {inp_vars[0]}->nb[{ggml_dim}]"
            nb1 = (
                f"{inp_vars[0]}->nb[1] * {step}"
                if ggml_dim == 1 and step != 1
                else f"{inp_vars[0]}->nb[1]"
            )
            nb2 = (
                f"{inp_vars[0]}->nb[2] * {step}"
                if ggml_dim == 2 and step != 1
                else f"{inp_vars[0]}->nb[2]"
            )
            nb3 = (
                f"{inp_vars[0]}->nb[3] * {step}"
                if ggml_dim == 3 and step != 1
                else f"{inp_vars[0]}->nb[3]"
            )
            lines.append(
                f"    tensors[{out_id}] = ggml_view_4d(ctx, {inp_vars[0]}, {', '.join(ne_strs)}, {nb1}, {nb2}, {nb3}, {offset});"
            )
            if out_id in self.graph.outputs:
                lines.append(
                    f"    if (!ggml_is_contiguous(tensors[{out_id}])) tensors[{out_id}] = ggml_cont(ctx, tensors[{out_id}]);"
                )
        elif node.opcode == GGMLOpCode.GGML_OP_CUSTOM_BIAS_GELU:
            lines.append("    #if defined(GGML_USE_CUDA)")
            lines.append(
                f"    struct ggml_tensor* b_{node.id} = {inp_vars[1]}; if (ggml_can_repeat(b_{node.id}, {inp_vars[0]})) b_{node.id} = ggml_repeat(ctx, b_{node.id}, {inp_vars[0]});"
            )
            lines.append(
                f"    tensors[{out_id}] = ggml_gelu(ctx, ggml_add(ctx, {inp_vars[0]}, b_{node.id}));"
            )
            lines.append("    #else")
            lines.append(
                f"    tensors[{out_id}] = ggml_map_custom2(ctx, {inp_vars[0]}, {inp_vars[1]}, ggmlc_compute_forward_bias_gelu, GGML_N_TASKS_MAX, nullptr);"
            )
            lines.append("    #endif")
        elif node.opcode == GGMLOpCode.GGML_OP_CUSTOM_LAYER_NORM:
            w_arg = inp_vars[1] if len(inp_vars) > 1 else "nullptr"
            b_arg = inp_vars[2] if len(inp_vars) > 2 else "nullptr"
            eps = node.attributes.get("eps", 1e-5)
            lines.append("    #if defined(GGML_USE_CUDA)")
            lines.append(f"    tensors[{out_id}] = ggml_norm(ctx, {inp_vars[0]}, {eps}f);")
            if w_arg != "nullptr":
                lines.append(
                    f"    if ({w_arg}) {{ struct ggml_tensor* w = {w_arg}; if (ggml_can_repeat(w, tensors[{out_id}])) w = ggml_repeat(ctx, w, tensors[{out_id}]); tensors[{out_id}] = ggml_mul(ctx, tensors[{out_id}], w); }}"
                )
            if b_arg != "nullptr":
                lines.append(
                    f"    if ({b_arg}) {{ struct ggml_tensor* b = {b_arg}; if (ggml_can_repeat(b, tensors[{out_id}])) b = ggml_repeat(ctx, b, tensors[{out_id}]); tensors[{out_id}] = ggml_add(ctx, tensors[{out_id}], b); }}"
                )
            lines.append("    #else")
            lines.append(
                f"    tensors[{out_id}] = ggml_map_custom3(ctx, {inp_vars[0]}, {w_arg}, {b_arg}, ggmlc_compute_forward_layer_norm, GGML_N_TASKS_MAX, nullptr);"
            )
            lines.append("    #endif")
        elif node.opcode == GGMLOpCode.GGML_OP_CUSTOM_RMS_NORM:
            w_arg = inp_vars[1] if len(inp_vars) > 1 else "nullptr"
            eps = node.attributes.get("eps", 1e-5)
            lines.append("    #if defined(GGML_USE_CUDA)")
            lines.append(f"    tensors[{out_id}] = ggml_rms_norm(ctx, {inp_vars[0]}, {eps}f);")
            if w_arg != "nullptr":
                lines.append(
                    f"    if ({w_arg}) {{ struct ggml_tensor* w = {w_arg}; if (ggml_can_repeat(w, tensors[{out_id}])) w = ggml_repeat(ctx, w, tensors[{out_id}]); tensors[{out_id}] = ggml_mul(ctx, tensors[{out_id}], w); }}"
                )
            lines.append("    #else")
            lines.append(
                f"    tensors[{out_id}] = ggml_map_custom2(ctx, {inp_vars[0]}, {w_arg}, ggmlc_compute_forward_rms_norm, GGML_N_TASKS_MAX, nullptr);"
            )
            lines.append("    #endif")
        elif node.opcode == GGMLOpCode.GGML_OP_CUSTOM_SWIGLU:
            if len(inp_vars) == 1:
                swapped = node.attributes.get("swapped", 0)
                fn = "ggml_swiglu_swapped" if swapped else "ggml_swiglu"
                lines.append(f"    tensors[{out_id}] = {fn}(ctx, {inp_vars[0]});")
            else:
                lines.append("    #if defined(GGML_USE_CUDA)")
                lines.append(
                    f"    tensors[{out_id}] = ggml_mul(ctx, ggml_silu(ctx, {inp_vars[0]}), {inp_vars[1]});"
                )
                lines.append("    #else")
                lines.append(
                    f"    tensors[{out_id}] = ggml_map_custom2(ctx, {inp_vars[0]}, {inp_vars[1]}, ggmlc_compute_forward_swiglu, GGML_N_TASKS_MAX, nullptr);"
                )
                lines.append("    #endif")
        elif node.opcode == GGMLOpCode.GGML_OP_PAD:
            p0 = node.attributes.get("pad_w", 0)
            p1 = node.attributes.get("pad_h", 0)
            p2 = node.attributes.get("pad_c", 0)
            p3 = node.attributes.get("pad_n", 0)
            lines.append(
                f"    tensors[{out_id}] = ggml_pad(ctx, {inp_vars[0]}, {p0}, {p1}, {p2}, {p3});"
            )
        elif node.opcode == GGMLOpCode.GGML_OP_SQR:
            exp = node.attributes.get("exponent", node.attributes.get("y", 2.0))
            lines.append(f"    struct ggml_tensor* sqr_in_{node.id} = {inp_vars[0]};")
            lines.append(
                f"    if (!ggml_is_contiguous(sqr_in_{node.id})) sqr_in_{node.id} = ggml_cont(ctx, sqr_in_{node.id});"
            )
            if float(exp) == 3.0:
                lines.append(
                    f"    tensors[{out_id}] = ggml_mul(ctx, sqr_in_{node.id}, ggml_sqr(ctx, sqr_in_{node.id}));"
                )
            else:
                lines.append(f"    tensors[{out_id}] = ggml_sqr(ctx, sqr_in_{node.id});")
        elif node.opcode in (
            GGMLOpCode.GGML_OP_SIN,
            GGMLOpCode.GGML_OP_COS,
            GGMLOpCode.GGML_OP_LOG,
        ):
            fn_name = {
                GGMLOpCode.GGML_OP_SIN: "ggml_sin",
                GGMLOpCode.GGML_OP_COS: "ggml_cos",
                GGMLOpCode.GGML_OP_LOG: "ggml_log",
            }[node.opcode]
            lines.append(f"    struct ggml_tensor* t_in_{node.id} = {inp_vars[0]};")
            lines.append(
                f"    if (!ggml_is_contiguous(t_in_{node.id})) t_in_{node.id} = ggml_cont(ctx, t_in_{node.id});"
            )
            lines.append(f"    tensors[{out_id}] = {fn_name}(ctx, t_in_{node.id});")
        elif node.opcode == GGMLOpCode.GGML_OP_GLU:
            lines.append(f"    tensors[{out_id}] = ggml_swiglu(ctx, {inp_vars[0]});")
        else:
            raise NotImplementedError(
                f"no C++ emission for {node.opcode.name} "
                f"(node {node.id}, '{node.name}'); "
                f"lower it to supported ops or add an emitter"
            )

        return lines

    def generate_main(self) -> str:
        """Generates ggmlc_main.cpp standalone execution entry point."""
        input_init_lines = []
        input_feed_lines = []
        for in_id in self.graph.inputs:
            t = self.graph.tensors[in_id]
            ident = _sanitize_ident(t.name)
            shape_str = ", ".join(_dim_to_cpp_expr(d) for d in t.ne)
            input_init_lines.append(
                f"    struct ggml_tensor* in_{ident} = ggml_new_tensor_4d(ctx, static_cast<enum ggml_type>({int(t.ggml_type)}), {shape_str});"
            )
            input_init_lines.append(f'    ggml_set_name(in_{ident}, "{t.name}");')
            input_init_lines.append(f'    inputs["{t.name}"] = in_{ident};')
            input_feed_lines.append(
                f"    if (in_{ident}) {{ std::vector<uint8_t> dummy_buf_{ident}(ggml_nbytes(in_{ident}), 0); ggml_backend_tensor_set(in_{ident}, dummy_buf_{ident}.data(), 0, ggml_nbytes(in_{ident})); }}"
            )

        if not input_init_lines:
            input_init_lines.append("    // No explicit model inputs declared")
        if not input_feed_lines:
            input_feed_lines.append("    // No input data buffers to initialize")

        inits_str = "\n".join(input_init_lines)
        feeds_str = "\n".join(input_feed_lines)

        return f"""// ============================================================================
// Standalone GGML Backend Model Runner: {self.model_name}
// Automatically generated by ggmlc. Supports CPU, CUDA, and Apple Metal.
// ============================================================================

#include "{self.model_name}.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <string>
#include <fstream>

int main(int argc, char** argv) {{
    std::string gguf_path = "{self.model_name}.gguf";
    if (std::ifstream f("model.gguf"); f.good()) {{
        gguf_path = "model.gguf";
    }}
    std::string device = "auto";
    int n_threads = 4;

    for (int i = 1; i < argc; ++i) {{
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {{
            gguf_path = argv[++i];
        }} else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {{
            n_threads = std::atoi(argv[++i]);
        }} else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {{
            device = argv[++i];
        }}
    }}

    std::cout << "=== Running {self.model_name} with GGML Backend ===" << std::endl;
    std::cout << "GGUF Model: " << gguf_path << " | Device: " << device << " | Threads: " << n_threads << std::endl;

    // 1. Initialize execution backend (CPU, CUDA, or Apple Metal)
    ggml_backend_t backend = nullptr;
#if defined(GGML_USE_CUDA)
    if (device == "auto" || device.rfind("cuda", 0) == 0) {{
        int dev_idx = 0;
        if (device.size() > 5 && device[4] == ':') {{
            dev_idx = std::stoi(device.substr(5));
        }}
        backend = ggml_backend_cuda_init(dev_idx);
        if (backend) {{
            std::cout << "Initialized GGML CUDA Backend on GPU " << dev_idx << std::endl;
        }}
    }}
#endif
#if defined(GGML_USE_METAL)
    if (!backend && (device == "auto" || device == "metal")) {{
        backend = ggml_backend_metal_init();
        if (backend) {{
            std::cout << "Initialized GGML Metal Backend" << std::endl;
        }}
    }}
#endif
    if (!backend) {{
        backend = ggml_backend_cpu_init();
        if (ggml_backend_is_cpu(backend)) {{
            ggml_backend_cpu_set_n_threads(backend, n_threads);
        }}
        std::cout << "Initialized GGML CPU Backend (" << n_threads << " threads)" << std::endl;
    }}

    // 2. Initialize GGML context for graph structures (no_alloc = true)
    struct ggml_init_params params = {{
        /*.mem_size   =*/ 128 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    }};
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {{
        std::cerr << "Failed to allocate GGML context" << std::endl;
        if (backend) ggml_backend_free(backend);
        return 1;
    }}

    // 3. Instantiate weights and register descriptors from GGUF metadata
    {self.model_name}::Weights weights;

    struct gguf_init_params gguf_params = {{ true, nullptr }};
    struct gguf_context* gguf_ctx = gguf_init_from_file(gguf_path.c_str(), gguf_params);
    if (!gguf_ctx) {{
        std::cerr << "Warning: Could not open GGUF file: " << gguf_path << " (proceeding with uninitialized weights)" << std::endl;
    }} else {{
        weights.init_tensors(ctx, gguf_ctx);
    }}

    // 4. Prepare input nodes
    std::unordered_map<std::string, struct ggml_tensor*> inputs;
{inits_str}

    // 5. Build computation graph
    struct ggml_cgraph* gf = {self.model_name}::build_graph(ctx, weights, inputs);

    // 6. Allocate memory on the target backend
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {{
        std::cerr << "Failed to allocate backend tensor memory" << std::endl;
        ggml_free(ctx);
        ggml_backend_free(backend);
        if (gguf_ctx) gguf_free(gguf_ctx);
        return 1;
    }}

    // 7. Load parameter tensor data from GGUF into backend buffer
    if (gguf_ctx) {{
        weights.load_data(gguf_ctx, gguf_path);
    }}

    // 8. Initialize input data via backend transfer
{feeds_str}

    // 9. Execute computation on target backend
    auto t0 = std::chrono::high_resolution_clock::now();
    enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (status == GGML_STATUS_SUCCESS) {{
        std::cout << "Execution completed successfully in " << elapsed_ms << " ms" << std::endl;
    }} else {{
        std::cerr << "Backend execution failed with status: " << status << std::endl;
    }}

    // Clean up
    if (gguf_ctx) gguf_free(gguf_ctx);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return (status == GGML_STATUS_SUCCESS) ? 0 : 1;
}}
"""

    def generate_cmakelists(self) -> str:
        """Generates CMakeLists.txt for compiling the generated C++ project."""
        return f"""cmake_minimum_required(VERSION 3.14)
project({self.model_name}_standalone LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_CUDA "Enable GGML CUDA GPU backend" OFF)
option(ENABLE_METAL "Enable GGML Metal GPU backend" OFF)

find_package(Threads REQUIRED)

find_package(ggml CONFIG QUIET)
if (ggml_FOUND)
    set(GGML_BACKEND_LIBS ggml::ggml Threads::Threads)
else()
    if (ENABLE_CUDA)
        enable_language(CUDA)
        add_compile_definitions(GGML_USE_CUDA)
        set(GGML_BACKEND_LIBS ggml ggml-base ggml-cpu ggml-cuda Threads::Threads)
    elseif (ENABLE_METAL)
        find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
        find_library(METAL_FRAMEWORK Metal REQUIRED)
        find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
        add_compile_definitions(GGML_USE_METAL)
        set(GGML_BACKEND_LIBS ggml ggml-base ggml-cpu ggml-metal ${{FOUNDATION_FRAMEWORK}} ${{METAL_FRAMEWORK}} ${{METALKIT_FRAMEWORK}} Threads::Threads)
    else()
        set(GGML_BACKEND_LIBS ggml ggml-base ggml-cpu Threads::Threads)
    endif()
endif()

add_executable({self.model_name}_run
    ggmlc_main.cpp
)

target_include_directories({self.model_name}_run PRIVATE
    ${{CMAKE_CURRENT_SOURCE_DIR}}
)

target_link_libraries({self.model_name}_run PRIVATE
    ${{GGML_BACKEND_LIBS}}
)
"""


def generate_cpp_project(
    graph: GGMLExecutionGraph,
    output_dir: str | Path,
    model_name: str = "model",
) -> dict[str, Path]:
    """Generates a complete standalone C++ project in output_dir."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    codegen = GGMLCCppCodeGenerator(graph, model_name=model_name)

    # 1. model.h
    header_path = out_p / f"{codegen.model_name}.h"
    header_path.write_text(codegen.generate_header(), encoding="utf-8")

    # 2. ggmlc_main.cpp
    main_path = out_p / "ggmlc_main.cpp"
    main_path.write_text(codegen.generate_main(), encoding="utf-8")

    # 3. CMakeLists.txt
    cmake_path = out_p / "CMakeLists.txt"
    cmake_path.write_text(codegen.generate_cmakelists(), encoding="utf-8")

    return {
        "header": header_path,
        "main": main_path,
        "cmake": cmake_path,
    }
