from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass
class DemoConfig:
    video_tokens: int = 16
    audio_tokens: int = 64
    model_dim: int = 32
    temperature: float = 0.2
    theta: float = 10_000.0
    seed: int = 7
    output_dir: str = "artifacts"


@dataclass
class DirectionMetrics:
    alignment_error: float
    accuracy: float
    peak_indices: list[int]
    predicted_labels: list[int]
    target_labels: list[int]


@dataclass
class VariantMetrics:
    name: str
    audio_rope_scale: float
    video_to_audio: DirectionMetrics
    audio_to_video: DirectionMetrics


SEGMENT_LABELS = ["speech", "drum", "bird", "engine", "speech", "bird", "drum", "engine"]
PALETTE = {
    "speech": "#ef476f",
    "drum": "#f9844a",
    "bird": "#43aa8b",
    "engine": "#577590",
    "grid": "#d5dce3",
    "text": "#1f2933",
    "background": "#ffffff",
}


def run_demo(config: DemoConfig) -> dict[str, object]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)
    base_query = _normalize([rng.gauss(0.0, 1.0) for _ in range(config.model_dim)])
    class_names = list(dict.fromkeys(SEGMENT_LABELS))
    label_index = {label: idx for idx, label in enumerate(class_names)}

    video_targets = _timeline_labels(config.video_tokens)
    audio_targets = _timeline_labels(config.audio_tokens)
    video_values = _one_hot_values(video_targets, label_index)
    audio_values = _one_hot_values(audio_targets, label_index)

    scaled = evaluate_variant(
        name="scaled",
        audio_rope_scale=config.video_tokens / config.audio_tokens,
        config=config,
        base_query=base_query,
        video_values=video_values,
        audio_values=audio_values,
        video_targets=video_targets,
        audio_targets=audio_targets,
    )
    unscaled = evaluate_variant(
        name="unscaled",
        audio_rope_scale=1.0,
        config=config,
        base_query=base_query,
        video_values=video_values,
        audio_values=audio_values,
        video_targets=video_targets,
        audio_targets=audio_targets,
    )

    summary = {
        "config": {
            "video_tokens": config.video_tokens,
            "audio_tokens": config.audio_tokens,
            "model_dim": config.model_dim,
            "temperature": config.temperature,
            "theta": config.theta,
            "seed": config.seed,
        },
        "class_names": class_names,
        "variants": [_serialize_variant(unscaled), _serialize_variant(scaled)],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for variant in (unscaled, scaled):
        _write_attention_svg(
            output_dir / f"{variant.name}_video_to_audio.svg",
            matrix=attention_matrix_for_variant(
                base_query=base_query,
                query_length=config.video_tokens,
                key_length=config.audio_tokens,
                query_scale=1.0,
                key_scale=variant.audio_rope_scale,
                temperature=config.temperature,
                theta=config.theta,
            ),
            row_labels=video_targets,
            column_labels=audio_targets,
            title=f"{variant.name.title()} video->audio attention",
            subtitle=(
                f"accuracy={variant.video_to_audio.accuracy:.3f}, "
                f"alignment_error={variant.video_to_audio.alignment_error:.2f}"
            ),
        )
        _write_attention_svg(
            output_dir / f"{variant.name}_audio_to_video.svg",
            matrix=attention_matrix_for_variant(
                base_query=base_query,
                query_length=config.audio_tokens,
                key_length=config.video_tokens,
                query_scale=variant.audio_rope_scale,
                key_scale=1.0,
                temperature=config.temperature,
                theta=config.theta,
            ),
            row_labels=audio_targets,
            column_labels=video_targets,
            title=f"{variant.name.title()} audio->video attention",
            subtitle=(
                f"accuracy={variant.audio_to_video.accuracy:.3f}, "
                f"alignment_error={variant.audio_to_video.alignment_error:.2f}"
            ),
        )

    _write_report(output_dir / "report.md", config, unscaled, scaled)
    return summary


def evaluate_variant(
    *,
    name: str,
    audio_rope_scale: float,
    config: DemoConfig,
    base_query: Sequence[float],
    video_values: Sequence[Sequence[float]],
    audio_values: Sequence[Sequence[float]],
    video_targets: Sequence[str],
    audio_targets: Sequence[str],
) -> VariantMetrics:
    video_to_audio = evaluate_direction(
        query_length=config.video_tokens,
        key_length=config.audio_tokens,
        query_scale=1.0,
        key_scale=audio_rope_scale,
        base_query=base_query,
        values=audio_values,
        query_targets=video_targets,
        key_targets=audio_targets,
        temperature=config.temperature,
        theta=config.theta,
    )
    audio_to_video = evaluate_direction(
        query_length=config.audio_tokens,
        key_length=config.video_tokens,
        query_scale=audio_rope_scale,
        key_scale=1.0,
        base_query=base_query,
        values=video_values,
        query_targets=audio_targets,
        key_targets=video_targets,
        temperature=config.temperature,
        theta=config.theta,
    )
    return VariantMetrics(
        name=name,
        audio_rope_scale=audio_rope_scale,
        video_to_audio=video_to_audio,
        audio_to_video=audio_to_video,
    )


def evaluate_direction(
    *,
    query_length: int,
    key_length: int,
    query_scale: float,
    key_scale: float,
    base_query: Sequence[float],
    values: Sequence[Sequence[float]],
    query_targets: Sequence[str],
    key_targets: Sequence[str],
    temperature: float,
    theta: float,
) -> DirectionMetrics:
    matrix = attention_matrix_for_variant(
        base_query=base_query,
        query_length=query_length,
        key_length=key_length,
        query_scale=query_scale,
        key_scale=key_scale,
        temperature=temperature,
        theta=theta,
    )
    class_names = list(dict.fromkeys(SEGMENT_LABELS))
    target_indices = [class_names.index(label) for label in query_targets]

    predicted_labels: list[int] = []
    peak_indices: list[int] = []
    correct = 0
    total_error = 0.0
    for row_index, weights in enumerate(matrix):
        fused = [0.0 for _ in values[0]]
        for weight, value in zip(weights, values):
            for class_index, class_value in enumerate(value):
                fused[class_index] += weight * class_value
        predicted = _argmax(fused)
        predicted_labels.append(predicted)
        peak = _argmax(weights)
        peak_indices.append(peak)
        if predicted == target_indices[row_index]:
            correct += 1
        expected = _expected_peak(row_index, query_length, key_length)
        total_error += abs(peak - expected)

    return DirectionMetrics(
        alignment_error=total_error / query_length,
        accuracy=correct / query_length,
        peak_indices=peak_indices,
        predicted_labels=predicted_labels,
        target_labels=target_indices,
    )


def attention_matrix_for_variant(
    *,
    base_query: Sequence[float],
    query_length: int,
    key_length: int,
    query_scale: float,
    key_scale: float,
    temperature: float,
    theta: float,
) -> list[list[float]]:
    queries = rope_encode(base_query, [index * query_scale for index in range(query_length)], theta)
    keys = rope_encode(base_query, [index * key_scale for index in range(key_length)], theta)
    matrix: list[list[float]] = []
    for query in queries:
        scores = [_dot(query, key) / temperature for key in keys]
        matrix.append(_softmax(scores))
    return matrix


def rope_encode(vector: Sequence[float], positions: Iterable[float], theta: float) -> list[list[float]]:
    if len(vector) % 2:
        raise ValueError("RoPE demo expects an even model dimension.")
    half = len(vector) // 2
    frequencies = [1.0 / (theta ** (i / half)) for i in range(half)]
    left = vector[:half]
    right = vector[half:]

    encoded: list[list[float]] = []
    for position in positions:
        rotated_left: list[float] = []
        rotated_right: list[float] = []
        for index, frequency in enumerate(frequencies):
            angle = position * frequency
            cosine = math.cos(angle)
            sine = math.sin(angle)
            x1 = left[index]
            x2 = right[index]
            rotated_left.append(x1 * cosine - x2 * sine)
            rotated_right.append(x1 * sine + x2 * cosine)
        encoded.append(rotated_left + rotated_right)
    return encoded


def _timeline_labels(length: int) -> list[str]:
    labels: list[str] = []
    segments = len(SEGMENT_LABELS)
    for token_index in range(length):
        segment_index = min(segments - 1, int(token_index * segments / length))
        labels.append(SEGMENT_LABELS[segment_index])
    return labels


def _one_hot_values(targets: Sequence[str], label_index: dict[str, int]) -> list[list[float]]:
    values: list[list[float]] = []
    width = len(label_index)
    for label in targets:
        row = [0.02] * width
        row[label_index[label]] = 1.0
        values.append(row)
    return values


def _expected_peak(query_index: int, query_length: int, key_length: int) -> int:
    if query_length == 1:
        return 0
    return round(query_index * (key_length - 1) / (query_length - 1))


def _serialize_variant(variant: VariantMetrics) -> dict[str, object]:
    return {
        "name": variant.name,
        "audio_rope_scale": variant.audio_rope_scale,
        "video_to_audio": {
            "alignment_error": variant.video_to_audio.alignment_error,
            "accuracy": variant.video_to_audio.accuracy,
            "peak_indices": variant.video_to_audio.peak_indices,
        },
        "audio_to_video": {
            "alignment_error": variant.audio_to_video.alignment_error,
            "accuracy": variant.audio_to_video.accuracy,
            "peak_indices": variant.audio_to_video.peak_indices,
        },
    }


def _write_report(path: Path, config: DemoConfig, unscaled: VariantMetrics, scaled: VariantMetrics) -> None:
    content = f"""# Ovi Demo Report

This toy demo recreates one narrow mechanism from Ovi: bidirectional cross-modal attention between
audio and video streams with RoPE-based temporal positions. The audio stream runs at
`{config.audio_tokens}` tokens while the video stream runs at `{config.video_tokens}` tokens. The only
change between the two variants is the audio RoPE scale.

## Result

- `unscaled` audio RoPE (`scale=1.0`) keeps the audio positions on their native faster grid.
- `scaled` audio RoPE (`scale={scaled.audio_rope_scale:.4f}`) compresses the audio grid to the
  video timeline, matching Ovi's central alignment trick.

### Video -> Audio retrieval

- Unscaled accuracy: `{unscaled.video_to_audio.accuracy:.3f}`
- Scaled accuracy: `{scaled.video_to_audio.accuracy:.3f}`
- Unscaled alignment error: `{unscaled.video_to_audio.alignment_error:.2f}` audio tokens
- Scaled alignment error: `{scaled.video_to_audio.alignment_error:.2f}` audio tokens

### Audio -> Video retrieval

- Unscaled accuracy: `{unscaled.audio_to_video.accuracy:.3f}`
- Scaled accuracy: `{scaled.audio_to_video.accuracy:.3f}`
- Unscaled alignment error: `{unscaled.audio_to_video.alignment_error:.2f}` video tokens
- Scaled alignment error: `{scaled.audio_to_video.alignment_error:.2f}` video tokens

## Faithful vs simplified

- Faithful to Ovi: twin branches, bidirectional cross-attention, mismatched temporal resolutions,
  and RoPE scaling to align them.
- Omitted on purpose: diffusion / flow matching, latent audio-video VAEs, the shared T5 text
  encoder, multimillion-sample training, and any learned weights.
"""
    path.write_text(content, encoding="utf-8")


def _write_attention_svg(
    path: Path,
    *,
    matrix: Sequence[Sequence[float]],
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    title: str,
    subtitle: str,
) -> None:
    cell = 12
    top = 64
    left = 96
    bottom = 40
    right = 20
    width = left + len(column_labels) * cell + right
    height = top + len(row_labels) * cell + bottom

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{PALETTE["background"]}"/>',
        f'<text x="{left}" y="24" font-family="monospace" font-size="18" fill="{PALETTE["text"]}">{_escape(title)}</text>',
        f'<text x="{left}" y="44" font-family="monospace" font-size="12" fill="{PALETTE["text"]}">{_escape(subtitle)}</text>',
    ]

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            x = left + column_index * cell
            y = top + row_index * cell
            color = _heat_color(value)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" '
                f'stroke="{PALETTE["grid"]}" stroke-width="0.4"/>'
            )

    for row_index, label in enumerate(row_labels):
        y = top + row_index * cell + cell - 3
        parts.append(
            f'<text x="8" y="{y}" font-family="monospace" font-size="10" fill="{PALETTE["text"]}">{_escape(label[:8])}</text>'
        )
    for column_index, label in enumerate(column_labels):
        x = left + column_index * cell + 3
        y = top - 4
        parts.append(
            f'<text transform="rotate(-60 {x} {y})" x="{x}" y="{y}" '
            f'font-family="monospace" font-size="8" fill="{PALETTE["text"]}">{_escape(label[:8])}</text>'
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _heat_color(value: float) -> str:
    clamped = max(0.0, min(1.0, value * 3.0))
    red = int(237 * clamped + 240 * (1.0 - clamped))
    green = int(71 * clamped + 245 * (1.0 - clamped))
    blue = int(111 * clamped + 249 * (1.0 - clamped))
    return f"#{red:02x}{green:02x}{blue:02x}"


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _argmax(values: Sequence[float]) -> int:
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def _normalize(values: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return [0.0 for _ in values]
    return [value / norm for value in values]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _softmax(values: Sequence[float]) -> list[float]:
    peak = max(values)
    exp_values = [math.exp(value - peak) for value in values]
    total = sum(exp_values)
    return [value / total for value in exp_values]

