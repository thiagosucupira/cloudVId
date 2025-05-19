# CloudVid

CloudVid is an AI-powered video generation tool that leverages Stable Diffusion models to transform text descriptions into beautiful video sequences with support for 1080p HD output.

## Overview

CloudVid uses a two-stage pipeline:
1. **Text-to-Image (T2I)**: Converts textual descriptions into high-quality images (1024x768)
2. **Image-to-Video (I2V)**: Transforms those static images into dynamic video clips with motion (1920x1080)

The tool is designed to process batches of scene descriptions from a JSON file, making it easy to generate multiple video sequences in a single run.

## Features

- 🖼️ Generate high-quality images from text descriptions
- 🎬 Transform still images into fluid video sequences 
- 🎛️ Fine-tune both image and video generation with customizable parameters
- 🔄 Process multiple scenes in batches for efficient workflow
- 🐳 Docker support for easy deployment
- 📺 Full HD 1080p video output
- 🚀 Optimized for cloud GPU environments

## Requirements

- NVIDIA GPU with CUDA support (recommended 16GB+ VRAM for 1080p)
- Python 3.8+
- PyTorch with CUDA
- Docker (optional)

## Installation

### Using Python

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Using Docker

1. Build the Docker image:
   ```
   docker build -t cloudvid .
   ```
2. Run the container:
   ```
   docker run --gpus all -v /path/to/input:/app/input_data -v /path/to/output:/app/output_data cloudvid
   ```

## Usage

1. Prepare your scene descriptions in a JSON file (see `input_scenes.json` for example format)
2. Run the processor:
   ```
   python process_video_batch.py --json_file input_scenes.json --output_dir output
   ```

### Input JSON Format

The input JSON is an array of scene objects, each containing:
- `id`: Unique identifier for the scene
- `scene_desc`: Text description for image generation
- `motion_desc`: Description of the motion (for documentation only)
- `t2i_params`: Parameters for text-to-image generation
- `i2v_params`: Parameters for image-to-video conversion

Example:
```json
[
  {
    "id": "dragon_flight_001",
    "scene_desc": "Epic high fantasy landscape, a majestic dragon with shimmering golden scales...",
    "motion_desc": "Smooth, slow, soaring flight, slight wing flaps.",
    "t2i_params": {
      "num_inference_steps": 30,
      "guidance_scale": 7.0,
      "width": 1024,
      "height": 768,
      "negative_prompt": "blurry, low quality, cartoon, watermark, text"
    },
    "i2v_params": {
      "motion_bucket_id": 100,
      "fps": 24,
      "num_frames": 25,
      "decode_chunk_size": 8,
      "noise_aug_strength": 0.02
    }
  }
]
```

## Output

For each scene, CloudVid generates:
- A base image file (`{scene_id}_base_image.png`)
- A 1080p video file (`{scene_id}_video.mp4`)

## Cloud Deployment

This tool is optimized for cloud GPU environments. Here are recommended options:

### Runpod.io (Most Cost-Effective)
- Create a GPU instance with RTX 4090 ($0.49/hour)
- Clone from GitHub: `git clone https://github.com/thiagosucupira/cloudVId.git`
- Build: `cd cloudVId && docker build -t cloudvid .`
- Run: `docker run --gpus all -v $(pwd)/input:/app/input_data -v $(pwd)/output:/app/output_data cloudvid`

### Google Colab
- Use Colab Pro/Pro+ for access to better GPUs
- Create a notebook with:
  ```python
  !git clone https://github.com/thiagosucupira/cloudVId.git
  %cd cloudVId
  !pip install -r requirements.txt
  !python process_video_batch.py --json_file input_scenes.json --output_dir output
  ```

## Models

The project uses the following models by default:
- Text-to-Image: `stabilityai/stable-diffusion-2-1-base`
- Image-to-Video: `stabilityai/stable-video-diffusion-img2vid-xt`

You can modify these in the `process_video_batch.py` file.

## HD Video Performance Tips

- **VRAM Requirements**: 1080p generation requires at least 16GB VRAM (24GB recommended)
- **Cloud GPUs**: Use RTX 4090 (24GB) on Runpod.io for optimal price/performance
- **Adjust Batch Size**: Modify `decode_chunk_size` (smaller values use less VRAM)
- **Model Selection**: For lower VRAM GPUs, consider using the non-XT version of the video model
- **xformers**: Enabled by default for memory-efficient attention

## Converting YouTube Video Creator Format

CloudVid now supports converting JSON files from the YouTube Video Creator format. The `convert_to_cloudvid.py` script transforms YouTube's scene descriptions into CloudVid-compatible format.

### How It Works

The converter:
- Extracts scene descriptions and motion directions from the YouTube format
- Creates proper t2i and i2v parameter sets for each scene
- Preserves aspect ratio information (e.g., `::1.3` notation in scene descriptions)
- Handles both regular scenes and nested event scenes
- Preserves all original data in an enriched output file

### Usage

```bash
python convert_to_cloudvid.py input_youtube_json [--output output_path] [--no-preserve]
```

Parameters:
- `input_youtube_json`: Path to the YouTube format JSON file
- `--output, -o`: (Optional) Output path for the CloudVid JSON file. If not specified, a timestamped file will be created in the same directory
- `--no-preserve`: (Optional) Do not create an enriched version with the original data

### Output Files

The converter creates two files:
1. A CloudVid-compatible JSON file that can be directly used with the `process_video_batch.py` script
2. An enriched version with `_with_original` suffix that preserves all the original YouTube data

### Example

```bash
# Basic conversion
python convert_to_cloudvid.py output/yout_20250501_011057/raw_json_yout_20250501_011057.json

# Specify output file
python convert_to_cloudvid.py input.json -o converted_scenes.json

# Convert without creating the enriched version
python convert_to_cloudvid.py input.json --no-preserve
```

### Format Compatibility

The YouTube Video Creator JSON has this structure:
```json
{
  "title": "Video Title",
  "art_style": "Description of art style",
  "scene_directions": [
    {
      "scene_desc": "Scene description",
      "motion_desc": "Motion description",
      "narration": "Narration text",
      // Other fields...
    },
    // More scenes...
  ]
}
```

It gets converted to CloudVid's format:
```json
[
  {
    "id": "scene_0",
    "scene_desc": "Scene description",
    "motion_desc": "Motion description",
    "t2i_params": {
      "num_inference_steps": 30,
      "guidance_scale": 7.5,
      "width": 768,
      "height": 512,
      "negative_prompt": "blurry, low quality, watermark, text, deformed"
    },
    "i2v_params": {
      "motion_bucket_id": 127,
      "fps": 8,
      "num_frames": 24,
      "decode_chunk_size": 8,
      "noise_aug_strength": 0.02
    },
    "metadata": {
      "original_data": {
        // All original scene data preserved here
      }
    }
  },
  // More scenes...
]
```

## License

This project is provided as-is for research and creative purposes.

## Acknowledgments

This project leverages the incredible work of Stability AI and the Hugging Face community.
