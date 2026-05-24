#!/bin/sh
set -eu

APP_DIR=${COMFYUI_APP_DIR:-/opt/ComfyUI}
DATA_DIR=${COMFYUI_DATA_DIR:-/data}
HOST_MODELS_DIR=${COMFYUI_MODELS_DIR:-/mnt/comfyui/models}
HOST_CUSTOM_NODES_DIR=${COMFYUI_CUSTOM_NODES_DIR:-/mnt/comfyui/custom_nodes}
EXTRA_PATHS_CONFIG=${COMFYUI_EXTRA_MODEL_PATHS_CONFIG:-/data/extra_model_paths.yaml}
COMFYUI_LISTEN=${COMFYUI_LISTEN:-0.0.0.0}

mkdir -p \
    "$DATA_DIR/input" \
    "$DATA_DIR/output" \
    "$DATA_DIR/temp" \
    "$DATA_DIR/user" \
    "$HOST_MODELS_DIR" \
    "$HOST_CUSTOM_NODES_DIR"

for path in "$DATA_DIR" "$DATA_DIR/input" "$DATA_DIR/output" "$DATA_DIR/temp" "$DATA_DIR/user"; do
    if [ ! -w "$path" ]; then
        echo "ComfyUI Docker error: '$path' is not writable by uid $(id -u)." >&2
        echo "Fix host bind ownership, for example: sudo chown -R \$(id -u):\$(id -g) input output temp user" >&2
        exit 1
    fi
done

touch "$DATA_DIR/.write-test" "$DATA_DIR/user/.write-test" "$DATA_DIR/user/comfyui.db.write-test"
rm -f "$DATA_DIR/.write-test" "$DATA_DIR/user/.write-test" "$DATA_DIR/user/comfyui.db.write-test"

cat > "$EXTRA_PATHS_CONFIG" <<EOF
host_models:
  base_path: ${HOST_MODELS_DIR}
  checkpoints: checkpoints
  text_encoders: |
    text_encoders
    clip
  clip_vision: clip_vision
  configs: configs
  controlnet: controlnet
  diffusion_models: |
    diffusion_models
    unet
  embeddings: embeddings
  loras: loras
  upscale_models: upscale_models
  vae: vae
  vae_approx: vae_approx
  gligen: gligen
  hypernetworks: hypernetworks
  photomaker: photomaker
  style_models: style_models
  model_patches: model_patches
  audio_encoders: audio_encoders
  background_removal: background_removal
  frame_interpolation: frame_interpolation
  geometry_estimation: geometry_estimation
  optical_flow: optical_flow
  classifiers: classifiers
  diffusers: diffusers
  latent_upscale_models: latent_upscale_models

host_custom_nodes:
  custom_nodes: ${HOST_CUSTOM_NODES_DIR}
EOF

cd "$APP_DIR"

# COMFYUI_ARGS is intentionally split by the shell so users can pass normal CLI flags.
# shellcheck disable=SC2086
exec python main.py \
    --listen "$COMFYUI_LISTEN" \
    --port 8188 \
    --input-directory "$DATA_DIR/input" \
    --output-directory "$DATA_DIR/output" \
    --temp-directory "$DATA_DIR" \
    --user-directory "$DATA_DIR/user" \
    --database-url "sqlite:///$DATA_DIR/user/comfyui.db" \
    --extra-model-paths-config "$EXTRA_PATHS_CONFIG" \
    --disable-auto-launch \
    ${COMFYUI_ARGS:-} \
    "$@"
