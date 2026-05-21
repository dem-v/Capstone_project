from __future__ import annotations

import torch


def model_probability(model: torch.nn.Module, image: torch.Tensor, class_idx: int) -> float:
    with torch.inference_mode():
        output = model(image)
        return float(torch.sigmoid(output[0, class_idx]).detach().cpu().item())


def faithfulness_baseline_tensor(model_input: torch.Tensor, baseline: str) -> torch.Tensor:
    if baseline == "zero_tensor":
        return torch.zeros_like(model_input)
    if baseline == "black":
        return torch.full_like(model_input, -1024.0)
    if baseline == "white":
        return torch.full_like(model_input, 1024.0)
    if baseline == "case_mean":
        return torch.full_like(model_input, float(model_input.mean().item()))
    raise ValueError(f"Unsupported faithfulness baseline: {baseline}")


def faithfulness_curve_rows(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    heatmap: torch.Tensor,
    class_idx: int,
    fractions: list[float],
    baseline: torch.Tensor,
) -> list[dict[str, float]]:
    if not fractions:
        return []
    flat_order = torch.argsort(heatmap.flatten().to(model_input.device), descending=True)
    original_flat = model_input.detach().clone().flatten()
    baseline_flat = baseline.detach().clone().flatten().to(model_input.device)
    rows: list[dict[str, float]] = []
    total_pixels = flat_order.numel()
    for fraction in fractions:
        keep_count = int(round(total_pixels * fraction))
        insertion_flat = baseline_flat.clone()
        deletion_flat = original_flat.clone()
        if keep_count > 0:
            selected = flat_order[:keep_count]
            insertion_flat[selected] = original_flat[selected]
            deletion_flat[selected] = baseline_flat[selected]
        insertion = insertion_flat.view_as(model_input)
        deletion = deletion_flat.view_as(model_input)
        rows.append(
            {
                "fraction": round(fraction, 6),
                "insertion_probability": round(
                    model_probability(model, insertion, class_idx), 6
                ),
                "deletion_probability": round(
                    model_probability(model, deletion, class_idx), 6
                ),
            }
        )
    return rows


def curve_auc(rows: list[dict[str, str | int | float]], value_key: str) -> float:
    points = sorted((float(row["fraction"]), float(row[value_key])) for row in rows)
    if len(points) < 2:
        return 0.0
    auc = 0.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        auc += (x1 - x0) * (y0 + y1) / 2.0
    return auc