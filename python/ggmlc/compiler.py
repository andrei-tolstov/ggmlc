"""Unified high-level compilation and code generation API for ggmlc."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ggmlc.codegen.cpp import generate_cpp_project
from ggmlc.dialect.ggml.lowering import GGMLExecutionGraph, lower_to_ggml
from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.quantization import quantize_graph_parameters
from ggmlc.transforms import create_standard_optimization_pipeline
from ggmlc.transforms.fusion import FusionOptions


@dataclass
class CompileOptions:
    """Structured configuration options for compiling models in ggmlc."""

    model_name: str = "model"
    enable_optimizations: bool = True
    enable_fusion: bool = True
    fusion_options: FusionOptions | dict[str, bool] | None = None
    quantize: str | DType | None = None
    dynamic_shapes: tuple[dict[int, Any], ...] | dict[str, Any] | None = None
    extra_metadata: dict[str, Any] | None = None
    tasks: str | list[str] | None = None


def compile_to_graph(
    model: Any,
    sample_inputs: tuple[Any, ...] | list[Any] | None = None,
    dynamic_shapes: tuple[dict[int, Any], ...] | dict[str, Any] | None = None,
    model_name: str = "model",
    enable_optimizations: bool = True,
    enable_fusion: bool = True,
    fusion_options: FusionOptions | dict[str, bool] | None = None,
    quantize: str | DType | None = None,
    release_module_storage: bool = False,
    **kwargs: Any,
) -> GGMLExecutionGraph:
    """Ingests, optimizes, and lowers a neural network model into a GGMLExecutionGraph.

    This function serves as the single authoritative frontend pipeline for both
    direct GGUF binary serialization (ggmlc.compile) and standalone C++ code
    generation (ggmlc.codegen), guaranteeing identical ingestion, canonical IR
    optimizations, and GGML lowering across both compilation targets.
    """
    opts = FusionOptions.from_dict(fusion_options)

    # 1. Ingest model into Canonical IR Graph
    canonical_graph: Graph
    if isinstance(model, Graph):
        canonical_graph = model
    elif hasattr(model, "main_graph") and isinstance(model.main_graph, Graph):
        canonical_graph = model.main_graph
    elif hasattr(model, "graph_module") or hasattr(model, "module"):  # ExportedProgram or nn.Module
        from ggmlc.frontend.pytorch import export_torch_model

        if sample_inputs is None:
            raise ValueError("sample_inputs must be provided when compiling a PyTorch model.")
        inputs_tuple = tuple(sample_inputs) if isinstance(sample_inputs, list) else sample_inputs
        exported = export_torch_model(
            model,
            inputs_tuple,
            dynamic_shapes=dynamic_shapes,
            model_name=model_name,
            enable_fusion=enable_fusion,
            fusion_options=opts,
            release_module_storage=release_module_storage,
        )
        canonical_graph = exported.main_graph
    elif hasattr(model, "eqns"):  # JAX ClosedJaxpr
        from ggmlc.frontend.jax.importer import import_jaxpr

        canonical_graph = import_jaxpr(model, graph_name=model_name)
    elif callable(model) and not hasattr(model, "parameters"):  # JAX function or callable
        import jax

        from ggmlc.frontend.jax import import_jaxpr

        if sample_inputs is None:
            raise ValueError("sample_inputs must be provided when compiling a JAX function.")
        inputs_tuple = tuple(sample_inputs) if isinstance(sample_inputs, list) else sample_inputs
        jaxpr = jax.make_jaxpr(model)(*inputs_tuple)
        import_kwargs = {k: v for k, v in kwargs.items() if k not in ("device", "n_threads")}
        canonical_graph = import_jaxpr(jaxpr, graph_name=model_name, **import_kwargs)
    else:
        # Fallback PyTorch export attempt
        from ggmlc.frontend.pytorch import export_torch_model

        if sample_inputs is None:
            raise ValueError("sample_inputs must be provided for model compilation.")
        inputs_tuple = tuple(sample_inputs) if isinstance(sample_inputs, list) else sample_inputs
        exported = export_torch_model(
            model,
            inputs_tuple,
            dynamic_shapes=dynamic_shapes,
            model_name=model_name,
            enable_fusion=enable_fusion,
            fusion_options=opts,
            release_module_storage=release_module_storage,
        )
        canonical_graph = exported.main_graph

    canonical_graph.name = model_name

    # 2. Run Canonical IR Graph Optimizations
    if enable_optimizations:
        opt_pipeline = create_standard_optimization_pipeline(
            enable_fusion=enable_fusion, options=opts
        )
        opt_result = opt_pipeline.run(canonical_graph)
        canonical_graph = opt_result.graph

    # 3. Lower to GGML Dialect
    ggml_graph = lower_to_ggml(
        canonical_graph,
        enable_fusion=enable_fusion,
        fusion_options=opts,
    )

    # 4. Apply Block Quantization (Optional)
    if quantize is not None:
        if release_module_storage:
            import gc

            for tensor in canonical_graph.tensors.values():
                tensor.data = None
            gc.collect()
        ggml_graph, _ = quantize_graph_parameters(ggml_graph, target_dtype=quantize)

    return ggml_graph


def compile(
    model: Any,
    sample_inputs: tuple[Any, ...] | list[Any] | None = None,
    output: str | Path | None = None,
    dynamic_shapes: tuple[dict[int, Any], ...] | dict[str, Any] | None = None,
    model_name: str = "model",
    enable_optimizations: bool = True,
    enable_fusion: bool = True,
    fusion_options: FusionOptions | dict[str, bool] | None = None,
    quantize: str | DType | None = None,
    return_runner: bool = False,
    pipeline: Any = None,
    tasks: str | list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Path | bytes | Any:
    """Compiles a PyTorch or JAX neural network model into a standard GGUF v3 binary container.

    This function ingests the source model into Canonical IR, runs standard optimization
    passes (constant folding, dead-code elimination, redundant cast pruning), performs
    target dialect lowering to GGML, applies optional parameter quantization (Q4_0, Q8_0),
    and serializes the result into a standardized GGUF v3 file.

    Args:
        model: PyTorch model (nn.Module or ExportedProgram), JAX callable, or IR Graph.
        sample_inputs: Sample input tensors matching the model's forward signature.
        output: Optional file path to stream the .gguf binary container directly to disk.
        dynamic_shapes: Dynamic shape specifications (e.g. torch.export.Dim).
        model_name: Identifier name embedded in graph and GGUF metadata.
        enable_optimizations: If True, applies standard IR graph optimization passes.
        enable_fusion: If True, lowers composite subgraphs into high-performance fused ops.
        fusion_options: Optional granular flags for specific fusion patterns.
        quantize: Optional quantization format ('f16', 'q4_0', 'q8_0', DType.F16, DType.Q4_0, DType.Q8_0).
        return_runner: If True, automatically loads and returns an instantiated ModelRunner.
        pipeline: Optional multimodal pipeline preprocessor (e.g. VisionPreprocessor or BPETokenizer)
            whose schema metadata will be embedded into GGUF headers.
        tasks: Explicit task or list of tasks supported by this model (e.g. 'classification',
            ['embedding', 'similarity'], or 'text-generation') embedded into 'ggmlc.tasks'.
        extra_metadata: Optional custom key-value metadata dictionary embedded into the GGUF file.
        **kwargs: Additional framework-specific keyword arguments.

    Returns:
        Path to the saved .gguf file (if output is provided), ModelRunner (if return_runner=True),
        or raw GGUF v3 bytes (if output is None and return_runner=False).

    Example:
        >>> import ggmlc, torch, torchvision.models as models
        >>> model = models.resnet18().eval()
        >>> x = torch.randn(1, 3, 224, 224)
        >>> # Stream directly to disk
        >>> model_path = ggmlc.compile(model, (x,), output="resnet18.gguf")
        >>> runner = ggmlc.load(model_path)
    """
    release_module_storage = bool(kwargs.pop("release_module_storage", False))

    ggml_graph = compile_to_graph(
        model=model,
        sample_inputs=sample_inputs,
        dynamic_shapes=dynamic_shapes,
        model_name=model_name,
        enable_optimizations=enable_optimizations,
        enable_fusion=enable_fusion,
        fusion_options=fusion_options,
        quantize=quantize,
        release_module_storage=release_module_storage,
        **kwargs,
    )

    # Extract metadata from pipeline and tasks if provided
    combined_metadata: dict[str, Any] = dict(extra_metadata or {})
    if pipeline is not None:
        if hasattr(pipeline, "to_gguf_metadata"):
            combined_metadata.update(pipeline.to_gguf_metadata())
        elif isinstance(pipeline, dict):
            combined_metadata.update(pipeline)

    if tasks:
        task_list = [tasks] if isinstance(tasks, str) else [str(t) for t in tasks]
        combined_metadata["ggmlc.tasks"] = task_list

    # Stream directly to file or serialize to memory
    from ggmlc.runtime.runner import load
    from ggmlc.serialization.gguf import save_to_gguf, serialize_to_gguf

    device = kwargs.get("device", "cpu")
    n_threads = kwargs.get("n_threads", 1)

    if output is not None:
        out_path = save_to_gguf(ggml_graph, output, extra_metadata=combined_metadata)
        if release_module_storage and not return_runner:
            import gc

            from ggmlc.frontend.pytorch.exporter import cleanup_released_storage
            from ggmlc.serialization.spill import cleanup_spills

            for tensor in getattr(ggml_graph, "tensors", {}).values():
                tensor.data = None
            del ggml_graph
            gc.collect()
            cleanup_released_storage()
            cleanup_spills()
        if return_runner:
            return load(out_path, n_threads=n_threads, device=device)
        return out_path

    if return_runner:
        gguf_bytes = serialize_to_gguf(ggml_graph, extra_metadata=combined_metadata)
        return load(gguf_bytes, n_threads=n_threads, device=device)

    return serialize_to_gguf(ggml_graph, extra_metadata=combined_metadata)


def compile_to_bytes(
    model: Any,
    sample_inputs: tuple[Any, ...] | list[Any] | None = None,
    dynamic_shapes: tuple[dict[int, Any], ...] | dict[str, Any] | None = None,
    model_name: str = "model",
    enable_optimizations: bool = True,
    enable_fusion: bool = True,
    fusion_options: FusionOptions | dict[str, bool] | None = None,
    quantize: str | DType | None = None,
    **kwargs: Any,
) -> bytes:
    """Compiles a model and returns in-memory GGUF v3 bytes."""
    res = compile(
        model=model,
        sample_inputs=sample_inputs,
        output=None,
        dynamic_shapes=dynamic_shapes,
        model_name=model_name,
        enable_optimizations=enable_optimizations,
        enable_fusion=enable_fusion,
        fusion_options=fusion_options,
        quantize=quantize,
        return_runner=False,
        **kwargs,
    )
    assert isinstance(res, bytes)
    return res


def codegen(
    model: Any,
    sample_inputs: tuple[Any, ...] | list[Any],
    output_dir: str | Path,
    dynamic_shapes: tuple[dict[int, Any], ...] | dict[str, Any] | None = None,
    model_name: str = "model",
    enable_optimizations: bool = True,
    enable_fusion: bool = True,
    fusion_options: FusionOptions | dict[str, bool] | None = None,
    quantize: str | DType | None = None,
    **kwargs: Any,
) -> Path:
    """Transpiles a neural network graph into a standalone, human-readable C++ project.

    Generates <ModelName>.h (weights & build_graph), ggmlc_main.cpp (runner),
    and CMakeLists.txt ready for native compilation.

    Args:
        model: PyTorch model, JAX callable, or IR Graph.
        sample_inputs: Sample input tensors matching model input signature.
        output_dir: Destination directory for the generated C++ project.
        dynamic_shapes: Optional dynamic shape constraints.
        model_name: Base class/file name identifier.
        enable_optimizations: If True, applies standard IR optimizations.
        enable_fusion: If True, lowers composite subgraphs to fused ops.
        fusion_options: Optional granular fusion options.
        quantize: Optional quantization format ('f16', 'q4_0', 'q8_0', etc.).
        **kwargs: Additional frontend export keyword arguments.

    Returns:
        Path to the generated project directory.
    """
    ggml_graph = compile_to_graph(
        model=model,
        sample_inputs=sample_inputs,
        dynamic_shapes=dynamic_shapes,
        model_name=model_name,
        enable_optimizations=enable_optimizations,
        enable_fusion=enable_fusion,
        fusion_options=fusion_options,
        quantize=quantize,
        **kwargs,
    )

    generate_cpp_project(
        graph=ggml_graph,
        output_dir=output_dir,
        model_name=model_name,
    )
    return Path(output_dir)
