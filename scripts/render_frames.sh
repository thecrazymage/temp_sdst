#!/bin/bash
set -uo pipefail

BLENDER_PATH="./objaverse_eval/blender-3.3.21-linux-x64/blender"
PYTHON_SCRIPT_PATH="./objaverse_eval/render_utils/blender_script.py"
PATH_TO_BACKGROUND_IMAGE="./objaverse_eval/assets/background.png"
ENV_MAP_PATH="./objaverse_eval/assets/studio_small_06_2k.hdr"

if (( $# < 3 )); then
    echo "Usage: bash render_frames.sh path/to/objaverse/evalation/folder path/to/output/folder <stage_name>"
    exit 1
fi

start_dir="$1"
output_dir="${2%/}/"
STAGE="$3"

move_models() {
    local source_parent="$1"
    local dest_parent="$2"
    
    mkdir -p "$dest_parent/models"
    
    for dir in "$source_parent"*/; do
        [[ -d "$dir" ]] || continue
        
        local source_dir="${dir%/}/model_stage_$STAGE"
        local dest_dir="$dest_parent/models/$(basename "${dir%/}")"
        
        if [[ -d "$source_dir" ]]; then
            mkdir -p "$dest_dir"
            cp -a "$source_dir/." "$dest_dir/"
            echo "Copied: $source_dir -> $dest_dir"
        else
            echo -e "\nSource directory $source_dir does not exist. Skipping.\n" >&2
        fi
    done
}

render_models() {
    local input_dir="$1"
    local output_dir="$2"
    local trajectory="$3"
    
    find "$input_dir" -type f -name "*.obj" | while IFS= read -r obj_file; do
        echo "Processing: $obj_file (trajectory: $trajectory)"
        "$BLENDER_PATH" -b -P "$PYTHON_SCRIPT_PATH" -- \
            --object_path "$obj_file" \
            --output_dir "$output_dir" \
            --engine "CYCLES" \
            --trajectory "$trajectory" \
            --camera_dist 1.4 \
            --env_map_strength 0.7 \
            --device OPTIX
    done
}

compile_gif() {
    local folder="$1"
    local output_folder="$2"
    local model_name=$(basename $folder)
    
    echo "Generating video: $model_name"
    ffmpeg -framerate 24 -pattern_type glob -i "$folder*.png" -i $PATH_TO_BACKGROUND_IMAGE -filter_complex "[1:v][0:v]overlay" -pix_fmt yuv420p -b 1200k "$output_folder/$model_name".mp4
}

traverse_directories_and_compile_gifs() {
    local dir="$1"
    local gif_destination_dir="$2"
    
    mkdir -p "$gif_destination_dir"
    
    for subdir in "$dir"/*/; do
        if [[ -d "$subdir" ]]; then
            compile_gif "$subdir" "$gif_destination_dir"
        fi
    done
}

echo "Step 1: Moving models..."
move_models "$start_dir" "$output_dir"

echo "Step 2: Rendering frames for FID..."
render_models "${output_dir}models" "$output_dir" "frames"

echo "Step 3: Rendering frames for video..."
render_models "${output_dir}models" "$output_dir" "video"

echo "Step 4: Compiling videos..."
traverse_directories_and_compile_gifs "${output_dir}video" "${output_dir}mp4"

echo "Done!"