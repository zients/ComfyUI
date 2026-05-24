#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import secrets
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    tomllib = None


DEFAULT_CHECKPOINT = "novaAnimeXL_ilV180.safetensors"
MAX_SEED = 0xffffffffffffffff


def write_stdout(message: str) -> None:
    sys.stdout.write(f"{message}\n")


def write_stderr(message: str) -> None:
    sys.stderr.write(f"{message}\n")


@dataclass(frozen=True)
class PromptSpec:
    name: str
    positive: str
    negative: str


@dataclass(frozen=True)
class LoraSpec:
    name: str
    strength_model: float
    strength_clip: float


@dataclass(frozen=True)
class Job:
    index: int
    checkpoint: str
    loras: tuple[LoraSpec, ...]
    prompt: PromptSpec
    seed: int
    prefix: str


def normalize_text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if value is None:
        if allow_empty:
            return ""
        raise RuntimeError(f"Missing required prompt field: {field}")
    if not isinstance(value, str):
        raise RuntimeError(f"Prompt field must be a string: {field}")

    text = re.sub(r"[ \t]+", " ", value.strip())
    text = re.sub(r"\n{3,}", "\n\n", text)
    if not text and not allow_empty:
        raise RuntimeError(f"Prompt field cannot be empty: {field}")
    return text


def slugify(value: str, *, fallback: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return (slug or fallback)[:max_length]


def load_prompt_file(path: Path) -> list[PromptSpec]:
    if tomllib is None:
        raise RuntimeError("TOML prompt files require Python 3.11 or newer")

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("Prompt file must contain a TOML table")

    defaults = data.get("defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        raise RuntimeError("[defaults] must be a TOML table")

    default_negative = normalize_text(defaults.get("negative", ""), field="defaults.negative", allow_empty=True)

    prompt_items = data.get("prompts")
    if not isinstance(prompt_items, list) or not prompt_items:
        raise RuntimeError("Prompt file must contain at least one [[prompts]] entry")

    prompts: list[PromptSpec] = []
    seen_names: set[str] = set()
    for index, item in enumerate(prompt_items, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(f"[[prompts]] entry {index} must be a TOML table")

        raw_name = item.get("name", f"prompt_{index:02d}")
        name = slugify(normalize_text(raw_name, field=f"prompts[{index}].name"), fallback=f"prompt_{index:02d}")
        if name in seen_names:
            raise RuntimeError(f"Duplicate prompt name after slug normalization: {name}")
        seen_names.add(name)

        positive = normalize_text(item.get("positive"), field=f"prompts[{index}].positive")
        negative = normalize_text(item.get("negative", default_negative), field=f"prompts[{index}].negative", allow_empty=True)
        prompts.append(PromptSpec(name=name, positive=positive, negative=negative))

    return prompts


def request_json(base_url: str, path: str, *, data: dict | None = None, timeout: float = 10) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    body = None
    headers = {}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_system_stats(base_url: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return request_json(base_url, "/system_stats", timeout=5)
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"/system_stats did not respond within {timeout:.0f}s: {last_error}")


def assert_cuda_device(base_url: str, timeout: float) -> None:
    stats = wait_for_system_stats(base_url, timeout)
    devices = stats.get("devices", [])
    if not devices:
        raise RuntimeError("No devices reported by /system_stats")

    device = devices[0]
    device_type = device.get("type")
    name = str(device.get("name", ""))
    write_stdout(f"system_stats: device={name} type={device_type}")
    if device_type != "cuda":
        raise RuntimeError("Image generation validation expects a CUDA device")


def get_checkpoint_names(base_url: str) -> list[str]:
    info = request_json(base_url, "/object_info/CheckpointLoaderSimple")
    ckpt_field = info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"]
    if not ckpt_field or not isinstance(ckpt_field[0], list):
        raise RuntimeError("Unexpected CheckpointLoaderSimple ckpt_name schema")
    return ckpt_field[0]


def get_lora_names(base_url: str) -> list[str]:
    info = request_json(base_url, "/object_info/LoraLoader")
    lora_field = info["LoraLoader"]["input"]["required"]["lora_name"]
    if not lora_field or not isinstance(lora_field[0], list):
        raise RuntimeError("Unexpected LoraLoader lora_name schema")
    return lora_field[0]


def validate_seed(seed: int, *, label: str) -> int:
    if seed < 0 or seed > MAX_SEED:
        raise RuntimeError(f"{label} must be between 0 and {MAX_SEED}")
    return seed


def load_seed_file(path: Path) -> list[int]:
    seeds: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            seed = int(line)
        except ValueError as exc:
            raise RuntimeError(f"Invalid seed at {path}:{line_number}: {line}") from exc
        seeds.append(validate_seed(seed, label=f"Seed at {path}:{line_number}"))
    return seeds


def build_seeds(count: int, *, seed: int | None, seed_file: Path | None) -> list[int]:
    if seed_file is not None and seed is not None:
        raise RuntimeError("--seed and --seed-file cannot be used together")

    if seed_file is not None:
        seeds = load_seed_file(seed_file)
        if len(seeds) != count:
            raise RuntimeError(f"Expected {count} seeds in {seed_file}, got {len(seeds)}")
        return seeds

    if seed is not None:
        validate_seed(seed, label="--seed")
        return [(seed + index) % (MAX_SEED + 1) for index in range(count)]

    return [secrets.randbelow(MAX_SEED + 1) for _ in range(count)]


def select_prompts(prompts: list[PromptSpec], count: int, order: str) -> list[PromptSpec]:
    if count < 1:
        raise RuntimeError("--count must be at least 1")
    if order == "cycle":
        return [prompts[index % len(prompts)] for index in range(count)]
    if order == "random":
        return [secrets.choice(prompts) for _ in range(count)]
    raise RuntimeError(f"Unsupported prompt order: {order}")


def validate_lora_strength(strength: float, *, label: str) -> float:
    if not -100.0 <= strength <= 100.0:
        raise RuntimeError(f"{label} must be between -100 and 100")
    return strength


def parse_optional_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def parse_lora_spec(value: str, *, default_strength_model: float, default_strength_clip: float) -> LoraSpec:
    raw = value.strip()
    if not raw:
        raise RuntimeError("--lora cannot be empty")

    name = raw
    strength_model = default_strength_model
    strength_clip = default_strength_clip

    parts = raw.rsplit(":", 2)
    if len(parts) == 2:
        parsed_strength_model = parse_optional_float(parts[1])
        if parsed_strength_model is not None and parts[0]:
            name = parts[0]
            strength_model = parsed_strength_model
    elif len(parts) == 3:
        parsed_strength_model = parse_optional_float(parts[1])
        parsed_strength_clip = parse_optional_float(parts[2])
        if parsed_strength_model is not None and parsed_strength_clip is not None and parts[0]:
            name = parts[0]
            strength_model = parsed_strength_model
            strength_clip = parsed_strength_clip
        elif parsed_strength_clip is not None and parts[0] and parts[1]:
            name = f"{parts[0]}:{parts[1]}"
            strength_model = parsed_strength_clip

    return LoraSpec(
        name=name,
        strength_model=validate_lora_strength(strength_model, label=f"--lora {name} model strength"),
        strength_clip=validate_lora_strength(strength_clip, label=f"--lora {name} clip strength"),
    )


def build_loras(
    lora_args: list[str] | None,
    *,
    default_strength_model: float,
    default_strength_clip: float,
) -> tuple[LoraSpec, ...]:
    if not lora_args:
        return ()

    validate_lora_strength(default_strength_model, label="--lora-strength-model")
    validate_lora_strength(default_strength_clip, label="--lora-strength-clip")
    return tuple(
        parse_lora_spec(
            lora_arg,
            default_strength_model=default_strength_model,
            default_strength_clip=default_strength_clip,
        )
        for lora_arg in lora_args
    )


def lora_records(loras: tuple[LoraSpec, ...]) -> list[dict]:
    return [
        {
            "name": lora.name,
            "strength_model": lora.strength_model,
            "strength_clip": lora.strength_clip,
        }
        for lora in loras
    ]


def describe_loras(loras: tuple[LoraSpec, ...]) -> str:
    return ", ".join(f"{lora.name}({lora.strength_model:g}/{lora.strength_clip:g})" for lora in loras)


def build_jobs(
    *,
    checkpoint: str,
    loras: tuple[LoraSpec, ...],
    prompts: list[PromptSpec],
    count: int,
    prompt_order: str,
    seeds: list[int],
    output_prefix: str,
) -> list[Job]:
    selected_prompts = select_prompts(prompts, count, prompt_order)
    checkpoint_stem = slugify(Path(checkpoint).stem, fallback="checkpoint", max_length=60)
    lora_stem = ""
    if loras:
        lora_names = [slugify(Path(lora.name).stem, fallback="lora", max_length=30) for lora in loras]
        lora_stem = "_lora-" + "_".join(lora_names)

    return [
        Job(
            index=index,
            checkpoint=checkpoint,
            loras=loras,
            prompt=prompt,
            seed=seed,
            prefix=f"{output_prefix}/{index:04d}_{checkpoint_stem}{lora_stem}_{prompt.name}",
        )
        for index, (prompt, seed) in enumerate(zip(selected_prompts, seeds), start=1)
    ]


def make_prompt(
    job: Job,
    *,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    width: int,
    height: int,
    batch_size: int,
    denoise: float,
) -> dict:
    model_ref: list[object] = ["4", 0]
    clip_ref: list[object] = ["4", 1]

    prompt = {
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": job.checkpoint},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"batch_size": batch_size, "height": height, "width": width},
        },
    }

    for node_number, lora in enumerate(job.loras, start=10):
        node_id = str(node_number)
        prompt[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "clip": clip_ref,
                "lora_name": lora.name,
                "model": model_ref,
                "strength_clip": lora.strength_clip,
                "strength_model": lora.strength_model,
            },
        }
        model_ref = [node_id, 0]
        clip_ref = [node_id, 1]

    prompt.update(
        {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": cfg,
                    "denoise": denoise,
                    "latent_image": ["5", 0],
                    "model": model_ref,
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "seed": job.seed,
                    "steps": steps,
                },
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": clip_ref, "text": job.prompt.positive},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": clip_ref, "text": job.prompt.negative},
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": job.prefix, "images": ["8", 0]},
            },
        }
    )
    return prompt


def queue_prompt(base_url: str, prompt: dict) -> str:
    response = request_json(base_url, "/prompt", data={"prompt": prompt}, timeout=30)
    prompt_id = response.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"/prompt did not return prompt_id: {response}")
    return prompt_id


def wait_for_prompt(base_url: str, prompt_id: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = request_json(base_url, f"/history/{prompt_id}", timeout=30)
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("completed") is True:
                return entry
            if status.get("status_str") == "error":
                raise RuntimeError(f"Prompt {prompt_id} failed: {status}")
        time.sleep(5)
    raise RuntimeError(f"Prompt {prompt_id} did not complete within {timeout:.0f}s")


def extract_output_images(history_entry: dict) -> list[dict]:
    images: list[dict] = []
    outputs = history_entry.get("outputs", {})
    if not isinstance(outputs, dict):
        return images

    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        for image in node_output.get("images", []):
            if isinstance(image, dict):
                images.append(image)
    return images


def manifest_record(
    job: Job,
    *,
    status: str,
    prompt_id: str | None,
    images: list[dict],
    error: str | None,
    args: argparse.Namespace,
) -> dict:
    return {
        "index": job.index,
        "status": status,
        "checkpoint": job.checkpoint,
        "loras": lora_records(job.loras),
        "prompt_name": job.prompt.name,
        "positive": job.prompt.positive,
        "negative": job.prompt.negative,
        "seed": job.seed,
        "prefix": job.prefix,
        "prompt_id": prompt_id,
        "images": images,
        "error": error,
        "settings": {
            "steps": args.steps,
            "cfg": args.cfg,
            "sampler": args.sampler,
            "scheduler": args.scheduler,
            "width": args.width,
            "height": args.height,
            "batch_size": args.batch_size,
            "denoise": args.denoise,
        },
    }


def write_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images through a running ComfyUI API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8188")
    parser.add_argument("--prompt-file", type=Path, required=True, help="TOML file with [[prompts]] entries.")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint name to use for this run. Default: {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--lora",
        action="append",
        default=None,
        metavar="NAME[:MODEL[:CLIP]]",
        help=(
            "LoRA name from models/loras. Can be used multiple times. "
            "Optionally append model and clip strengths, for example: style.safetensors:0.8:0.8."
        ),
    )
    parser.add_argument(
        "--lora-strength-model",
        type=float,
        default=1.0,
        help="Default model strength for --lora entries without an inline model strength.",
    )
    parser.add_argument(
        "--lora-strength-clip",
        type=float,
        default=1.0,
        help="Default CLIP strength for --lora entries without an inline CLIP strength.",
    )
    parser.add_argument("--count", type=int, default=None, help="Total images to generate. Defaults to the number of prompts.")
    parser.add_argument("--prompt-order", choices=["cycle", "random"], default="cycle")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--run-name", default=None, help="Output run folder name. Defaults to a timestamp.")
    parser.add_argument("--skip-cuda-check", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately when a job fails.")
    parser.add_argument("--system-timeout", type=float, default=120)
    parser.add_argument("--timeout", type=float, default=900, help="Timeout per image request in seconds.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=6.5)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="normal")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--denoise", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None, help="First seed. Later jobs use consecutive seeds.")
    parser.add_argument("--seed-file", type=Path, default=None, help="Text file with one seed per job.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompts = load_prompt_file(args.prompt_file)
    count = args.count or len(prompts)
    seeds = build_seeds(count, seed=args.seed, seed_file=args.seed_file)
    loras = build_loras(
        args.lora,
        default_strength_model=args.lora_strength_model,
        default_strength_clip=args.lora_strength_clip,
    )

    run_name = args.run_name or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = slugify(run_name, fallback="run")
    output_prefix = f"generate_images/{run_name}"
    run_dir = args.output_dir / output_prefix
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.jsonl"
    if manifest_path.exists():
        raise RuntimeError(f"Manifest already exists, choose a different --run-name: {manifest_path}")

    if args.skip_cuda_check:
        wait_for_system_stats(args.base_url, args.system_timeout)
    else:
        assert_cuda_device(args.base_url, args.system_timeout)

    available_checkpoints = set(get_checkpoint_names(args.base_url))
    if args.checkpoint not in available_checkpoints:
        raise RuntimeError(f"Missing checkpoint: {args.checkpoint}")

    if loras:
        available_loras = set(get_lora_names(args.base_url))
        missing_loras = [lora.name for lora in loras if lora.name not in available_loras]
        if missing_loras:
            raise RuntimeError(f"Missing LoRA: {', '.join(missing_loras)}")

    jobs = build_jobs(
        checkpoint=args.checkpoint,
        loras=loras,
        prompts=prompts,
        count=count,
        prompt_order=args.prompt_order,
        seeds=seeds,
        output_prefix=output_prefix,
    )

    write_stdout(f"run_dir: {run_dir}")
    write_stdout(f"manifest: {manifest_path}")
    write_stdout("planned_jobs:")
    for job in jobs:
        write_stdout(
            json.dumps(
                {
                    "index": job.index,
                    "checkpoint": job.checkpoint,
                    "loras": lora_records(job.loras),
                    "prompt_name": job.prompt.name,
                    "seed": job.seed,
                    "prefix": job.prefix,
                },
                ensure_ascii=False,
            )
        )

    completed = 0
    failed = 0
    for job in jobs:
        lora_text = f" loras={describe_loras(job.loras)}" if job.loras else ""
        write_stdout(
            f"queueing {job.index}/{len(jobs)} checkpoint={job.checkpoint}{lora_text} "
            f"prompt={job.prompt.name} seed={job.seed}"
        )
        prompt_id: str | None = None
        try:
            prompt_id = queue_prompt(
                args.base_url,
                make_prompt(
                    job,
                    steps=args.steps,
                    cfg=args.cfg,
                    sampler=args.sampler,
                    scheduler=args.scheduler,
                    width=args.width,
                    height=args.height,
                    batch_size=args.batch_size,
                    denoise=args.denoise,
                ),
            )
            history_entry = wait_for_prompt(args.base_url, prompt_id, args.timeout)
            images = extract_output_images(history_entry)
            write_jsonl(
                manifest_path,
                manifest_record(job, status="completed", prompt_id=prompt_id, images=images, error=None, args=args),
            )
            completed += 1
            write_stdout(f"completed {job.index}/{len(jobs)} prompt_id={prompt_id}")
        except Exception as exc:
            failed += 1
            error = str(exc)
            write_jsonl(
                manifest_path,
                manifest_record(job, status="failed", prompt_id=prompt_id, images=[], error=error, args=args),
            )
            write_stderr(f"failed {job.index}/{len(jobs)}: {error}")
            if args.stop_on_error:
                break

    write_stdout(f"Finished: completed={completed} failed={failed} manifest={manifest_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        write_stderr(f"generation failed: {exc}")
        raise SystemExit(1)
