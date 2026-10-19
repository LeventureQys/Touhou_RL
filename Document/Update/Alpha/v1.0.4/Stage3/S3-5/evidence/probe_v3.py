"""S3-5 shape / parameter / gradient evidence for the v3-st model.

Companion to ``startup_v3.py``: that one proves the model can be built and
trained end to end, this one prints the *layer by layer* numbers the S3-5 report
quotes, so every shape claim in ``完成报告.md`` has a rerunnable source:

    A. declared tensor shapes (F4, F8, heatmap, danger, local patch/mask, fusion)
    B. per-module output shapes and parameter counts
    C. local-crop geometry (bin centres, 128 px window, mask centre/edge)
    D. gradient routing (heatmap loss vs policy loss, per group)
    E. ``center_detach`` and ``local_enabled`` ablations of that routing
    F. CPU forward wall time (forward only - NOT a training throughput claim)

Run from the workspace root with ``PYTHONPATH`` set::

    & $py Document/Update/Alpha/v1.0.4/Stage3/S3-5/evidence/probe_v3.py

CPU only, deterministic seeds, exits 0 when every assertion holds.
"""

from __future__ import annotations

import json
import time

import torch
import torch.nn as nn

from touhou_rl import obs
from touhou_rl.vmodels3 import (
    FUSED_VEC,
    LOCAL_VEC,
    NATIVE_PLAYFIELD_HW,
    VisionSpatialTemporalConfig,
    VisionSpatialTemporalPolicy,
    decode_location,
    local_grid,
    sample_local,
)

BATCH = 1
SEED = 5


def _text(shape) -> str:
    return "x".join(str(size) for size in shape)


def _tensor_row(name: str, tensor: torch.Tensor) -> dict:
    return {
        "tensor": name,
        "shape": _text(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": bool(torch.isfinite(tensor).all()),
    }


def section_a(model, config, batch) -> list[dict]:
    """Explicit shapes of every declared stage, including the ones that are not modules."""
    rows: list[dict] = []
    playfield = batch["playfield"].permute(0, 1, 4, 2, 3).reshape(BATCH, 9, 224, 192)
    rows.append(_tensor_row("input.playfield[B,3*depth,H,W]", playfield))
    rows.append(
        _tensor_row(
            "input.hud_score newest frame[B,3,H,W]",
            batch["hud_score"][:, -1].permute(0, 3, 1, 2),
        )
    )
    rows.append(
        _tensor_row(
            "input.hud_status newest frame[B,3,H,W]",
            batch["hud_status"][:, -1].permute(0, 3, 1, 2),
        )
    )

    f4, f8 = model.playfield(playfield)
    rows.append(_tensor_row("playfield.F4", f4))
    rows.append(_tensor_row("playfield.F8", f8))
    heatmap = model.locate(f4, f8)
    rows.append(_tensor_row("locate.heatmap_logits", heatmap))
    position, stats = decode_location(heatmap)
    rows.append(_tensor_row("decode.position", position))
    rows.append(_tensor_row("decode.location_stats", stats))
    location = model.location_mlp(torch.cat([position, stats], dim=-1))
    rows.append(_tensor_row("location_mlp.out", location))
    patch, validity = sample_local(f4, position.detach(), config)
    rows.append(_tensor_row("local.patch", patch))
    rows.append(_tensor_row("local.validity", validity))
    local_vector = model.local_features(f4, position)
    rows.append(_tensor_row("local.vector", local_vector))
    global_vector = model.global_features(f8)
    rows.append(_tensor_row("global.vector", global_vector))
    hud_score = model.hud_score(batch["hud_score"][:, -1].permute(0, 3, 1, 2))
    hud_status = model.hud_status(batch["hud_status"][:, -1].permute(0, 3, 1, 2))
    rows.append(_tensor_row("hud_score.vector", hud_score))
    rows.append(_tensor_row("hud_status.vector", hud_status))
    fused = torch.cat([local_vector, global_vector, hud_score, hud_status, location], dim=-1)
    rows.append(_tensor_row("fusion.input", fused))
    features = model.trunk(fused)
    rows.append(_tensor_row("trunk.features", features))
    rows.append(_tensor_row("policy_head.policy_logits", model.policy_head(features)))
    self_pred = torch.cat([position, model.self_head(features)], dim=-1)
    rows.append(_tensor_row("self_head.self_pred", self_pred))
    rows.append(_tensor_row("danger.danger_log", model.danger(f8)))
    assert fused.size(1) == FUSED_VEC == 400
    assert local_vector.size(1) == LOCAL_VEC
    return rows


def section_b(model) -> list[dict]:
    """Every conv/linear/pool module: output shape on a real forward + parameters."""
    records: list[dict] = []
    handles = []

    def hook(name: str, module: nn.Module):
        def record(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            records.append(
                {
                    "module": name,
                    "type": type(module).__name__,
                    "out_shape": _text(tensor.shape),
                    "parameters": sum(parameter.numel() for parameter in module.parameters()),
                }
            )

        return record

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
            handles.append(module.register_forward_hook(hook(name, module)))
    batch = _batch(model.config)
    with torch.no_grad():
        model(batch)
    for handle in handles:
        handle.remove()
    return records


def section_c(config) -> dict:
    grid = local_grid(torch.tensor([[0.5, 0.5]]), config)
    native_h, native_w = NATIVE_PLAYFIELD_HW
    offsets_x = (grid[0, 0, :, 0] - (2 * 0.5 - 1)) * native_w / 2
    offsets_y = (grid[0, :, 0, 1] - (2 * 0.5 - 1)) * native_h / 2
    ones = torch.ones(1, 1, 56, 48)
    _patch, centre_mask = sample_local(ones, torch.tensor([[0.5, 0.5]]), config)
    _patch, corner_mask = sample_local(ones, torch.tensor([[0.5 / 48, 0.5 / 56]]), config)
    return {
        "roi_native_hw": _text(config.roi_native_hw),
        "local_grid_hw": _text(config.local_grid_hw),
        "bin_centres_x_native_px": [round(float(value), 3) for value in offsets_x],
        "bin_centres_y_native_px": [round(float(value), 3) for value in offsets_y],
        "bin_pitch_native_px": round(float(offsets_x[1] - offsets_x[0]), 3),
        "window_span_native_px": round(float(offsets_x[-1] - offsets_x[0]) + 8.0, 3),
        "mask_centre_min": round(float(centre_mask.min()), 6),
        "mask_centre_max": round(float(centre_mask.max()), 6),
        "mask_corner_min": round(float(corner_mask.min()), 6),
        "mask_corner_max": round(float(corner_mask.max()), 6),
        "grid_centre_column_x": round(float(grid[0, 0, 0, 0]), 6),
    }


def _grad_groups(model) -> dict[str, list[torch.nn.Parameter]]:
    return {
        "playfield_trunk": list(model.playfield.parameters()),
        "heat_conv": list(model.heat_conv.parameters()),
        "heatmap_head": list(model.heatmap_head.parameters()),
        "location_mlp": list(model.location_mlp.parameters()),
        "local": list(model.local.parameters()),
        "global": list(model.global_conv.parameters()) + list(model.global_fc.parameters()),
        "hud_score": list(model.hud_score.parameters()),
        "hud_status": list(model.hud_status.parameters()),
        "trunk": list(model.trunk.parameters()),
        "policy_head": list(model.policy_head.parameters()),
        "self_head": list(model.self_head.parameters()),
        "danger": list(model.danger.parameters()),
    }


def _grad_table(model) -> dict[str, str]:
    table = {}
    for name, parameters in _grad_groups(model).items():
        total = 0.0
        touched = False
        for parameter in parameters:
            if parameter.grad is None:
                continue
            touched = True
            total += float(parameter.grad.detach().abs().sum())
        table[name] = ("None" if not touched else f"{total:.6f}")
    return table


def section_d(config) -> dict:
    heatmap_loss, policy_loss = {}, {}
    torch.manual_seed(11)
    model = VisionSpatialTemporalPolicy(config).eval()
    model(_batch(config))["heatmap_logits"].sum().backward()
    heatmap_loss = _grad_table(model)
    model.zero_grad(set_to_none=True)
    torch.manual_seed(11)
    model = VisionSpatialTemporalPolicy(config).eval()
    model(_batch(config))["policy_logits"].sum().backward()
    policy_loss = _grad_table(model)
    return {"heatmap_loss": heatmap_loss, "policy_loss": policy_loss}


def section_e(config) -> dict:
    torch.manual_seed(12)
    model = VisionSpatialTemporalPolicy(config).eval()
    f4 = torch.rand(1, 64, 56, 48)

    position = torch.tensor([[0.4, 0.6]], requires_grad=True)
    model.local_features(f4, position).sum().backward()
    detached_grad = None if position.grad is None else float(position.grad.abs().sum())

    attached = VisionSpatialTemporalPolicy(
        VisionSpatialTemporalConfig(center_detach=False)
    ).eval()
    attached.load_state_dict(model.state_dict())
    position = torch.tensor([[0.4, 0.6]], requires_grad=True)
    attached.local_features(f4, position).sum().backward()
    attached_grad = None if position.grad is None else float(position.grad.abs().sum())

    disabled = VisionSpatialTemporalPolicy(
        VisionSpatialTemporalConfig(local_enabled=False)
    ).eval()
    disabled.load_state_dict(model.state_dict())
    vector = disabled.local_features(f4, torch.tensor([[0.5, 0.5]]))
    return {
        "parameter_count_enabled": model.num_parameters(),
        "parameter_count_disabled": disabled.num_parameters(),
        "state_dict_keys_identical": list(model.state_dict()) == list(disabled.state_dict()),
        "disabled_local_vector_nonzero": int(torch.count_nonzero(vector)),
        "disabled_local_vector_shape": _text(vector.shape),
        "detached_position_grad_sum": detached_grad,
        "attached_position_grad_sum": (
            None if attached_grad is None else round(attached_grad, 8)
        ),
    }


def _batch(config, batch: int = BATCH) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    return {
        name: torch.rand((batch, config.depth, height, width, 3), generator=generator)
        for name, (height, width) in config.region_sizes.items()
    }


def section_f(config) -> dict:
    torch.manual_seed(13)
    model = VisionSpatialTemporalPolicy(config).eval()
    timings = {}
    for batch_size in (1, 8):
        inputs = _batch(config, batch=batch_size)
        with torch.no_grad():
            model(inputs)  # warm-up
            started = time.perf_counter()
            for _ in range(3):
                model(inputs)
            seconds = (time.perf_counter() - started) / 3
        timings[f"batch_{batch_size}_seconds"] = round(seconds, 5)
        timings[f"batch_{batch_size}_ms_per_frame"] = round(1000.0 * seconds / batch_size, 3)
    timings["note"] = "CPU forward only; not a training throughput claim (S3-7 measures that)"
    return timings


def main() -> int:
    torch.manual_seed(SEED)
    config = VisionSpatialTemporalConfig()
    model = VisionSpatialTemporalPolicy(config).eval()
    batch = _batch(config)
    with torch.no_grad():
        declared = section_a(model, config, batch)
    modules = section_b(model)
    top_level = {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, module in model.named_children()
    }
    total = model.num_parameters()
    assert total == sum(top_level.values())
    report = {
        "seed": SEED,
        "batch": BATCH,
        "parameters": total,
        "parameters_by_child": top_level,
        "parameters_v2_reference": 554_873,
        "parameter_ratio_to_v2": round(total / 554_873, 4),
        "A_declared_tensors": declared,
        "B_modules": modules,
        "C_local_geometry": section_c(config),
        "D_gradient_routing": section_d(config),
        "E_ablations": section_e(config),
        "F_cpu_forward": section_f(config),
        "fusion_width": FUSED_VEC,
        "danger_cells": obs.DANGER_ROWS * obs.DANGER_COLS,
    }
    # The declared shapes are the contract: assert them here so the evidence file
    # cannot drift from the text.
    expected = {
        "playfield.F4": "1x64x56x48",
        "playfield.F8": "1x96x28x24",
        "locate.heatmap_logits": "1x56x48",
        "decode.position": "1x2",
        "decode.location_stats": "1x2",
        "location_mlp.out": "1x16",
        "local.patch": "1x64x16x16",
        "local.validity": "1x1x16x16",
        "local.vector": "1x128",
        "global.vector": "1x128",
        "hud_score.vector": "1x64",
        "hud_status.vector": "1x64",
        "fusion.input": "1x400",
        "trunk.features": "1x256",
        "policy_head.policy_logits": "1x18",
        "self_head.self_pred": "1x24",
        "danger.danger_log": "1x128",
        "input.playfield[B,3*depth,H,W]": "1x9x224x192",
    }
    observed = {row["tensor"]: row["shape"] for row in declared}
    for name, shape in expected.items():
        assert observed[name] == shape, (name, observed[name], shape)
    assert report["C_local_geometry"]["bin_pitch_native_px"] == 8.0
    assert report["C_local_geometry"]["window_span_native_px"] == 128.0
    assert report["C_local_geometry"]["mask_centre_min"] == 1.0
    assert report["C_local_geometry"]["mask_corner_min"] == 0.0
    assert report["E_ablations"]["disabled_local_vector_nonzero"] == 0
    assert (
        report["E_ablations"]["parameter_count_disabled"]
        == report["E_ablations"]["parameter_count_enabled"]
    )
    assert report["E_ablations"]["detached_position_grad_sum"] is None
    assert report["E_ablations"]["attached_position_grad_sum"] not in (None, 0.0)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    import sys

    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        sys.exit(1)
