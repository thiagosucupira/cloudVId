# CloudVid

CloudVid is an AI-powered video generation tool that leverages Stable Diffusion models to transform text descriptions into beautiful video sequences.

## Overview

CloudVid uses a two-stage pipeline:
1. **Text-to-Image (T2I)**: Converts textual descriptions into high-quality images
2. **Image-to-Video (I2V)**: Transforms those static images into dynamic video clips with motion

The tool is designed to process batches of scene descriptions from a JSON file, making it easy to generate multiple video sequences in a single run.

## Features

- 🖼️ Generate high-quality images from text descriptions
- 🎬 Transform still images into fluid video sequences
- 🎛️ Fine-tune both image and video generation with customizable parameters
- 🔄 Process multiple scenes in batches for efficient workflow
- 🐳 Docker support for easy deployment

## Requirements

- NVIDIA GPU with CUDA support (recommended)
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
      "width": 768,
      "height": 512,
      "negative_prompt": "blurry, low quality, cartoon, watermark, text"
    },
    "i2v_params": {
      "motion_bucket_id": 100,
      "fps": 6,
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
- A video file (`{scene_id}_video.mp4`)

## Models

The project uses the following models by default:
- Text-to-Image: `stabilityai/stable-diffusion-2-1-base`
- Image-to-Video: `stabilityai/stable-video-diffusion-img2vid-xt`

You can modify these in the `process_video_batch.py` file.

## Performance Tips

- Adjust `decode_chunk_size` based on available VRAM
- For lower VRAM GPUs, consider using the non-XT version of the video model
- Enable xformers for memory-efficient attention if installed

## License

This project is provided as-is for research and creative purposes.

## Acknowledgments

This project leverages the incredible work of Stability AI and the Hugging Face community. 