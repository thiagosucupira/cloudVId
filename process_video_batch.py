import json
import os
import argparse
import torch
from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
import time

# --- Model Configuration ---
# Default models - will be overridden by JSON input if provided
DEFAULT_T2I_MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
DEFAULT_I2V_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"

# --- Global Variables for Models (to load them once) ---
t2i_pipe = None
i2v_pipe = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_models(t2i_model_id=None, i2v_model_id=None):
    global t2i_pipe, i2v_pipe
    
    # Use provided model IDs or defaults
    t2i_model = t2i_model_id or DEFAULT_T2I_MODEL_ID
    i2v_model = i2v_model_id or DEFAULT_I2V_MODEL_ID
    
    print(f"Using device: {device}")
    print(f"Loading Text-to-Image model: {t2i_model}")
    t2i_pipe = StableDiffusionPipeline.from_pretrained(
        t2i_model,
        torch_dtype=torch.float16, # Use float16 for faster inference and less VRAM
        variant="fp16", # if available for the model
        use_safetensors=True
    )
    t2i_pipe.to(device)
    # Optional: Enable if xformers is installed and you want to try it
    # try:
    #     t2i_pipe.enable_xformers_memory_efficient_attention()
    #     print("xformers enabled for T2I.")
    # except ImportError:
    #     print("xformers not available or not compatible for T2I.")

    print(f"Loading Image-to-Video model: {i2v_model}")
    i2v_pipe = StableVideoDiffusionPipeline.from_pretrained(
        i2v_model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    i2v_pipe.to(device)
    # Optional: Enable if xformers is installed
    # try:
    #     i2v_pipe.enable_xformers_memory_efficient_attention()
    #     print("xformers enabled for I2V.")
    # except ImportError:
    #     print("xformers not available or not compatible for I2V.")

    print("Models loaded.")

def generate_image_from_text(prompt, negative_prompt=None, num_inference_steps=25, guidance_scale=7.5, width=512, height=512):
    print(f"  Generating image for prompt: '{prompt[:50]}...'")
    if t2i_pipe is None:
        raise RuntimeError("Text-to-Image model not loaded.")

    image = t2i_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height
    ).images[0]
    return image

def generate_video_from_image(image, motion_bucket_id=127, fps=7, num_frames=25, decode_chunk_size=8, noise_aug_strength=0.02):
    print(f"  Generating video from image...")
    if i2v_pipe is None:
        raise RuntimeError("Image-to-Video model not loaded.")

    # SVD expects a specific image size, often 1024x576 or 576x1024.
    # For higher quality, we'll use 1920x1080 if it can handle it, otherwise keep 1024x576
    original_width, original_height = image.size
    target_width, target_height = (1920, 1080)  # Target 1080p resolution
    
    # If SVD can't handle 1080p, fall back to its preferred resolution
    try:
        if original_width != target_width or original_height != target_height:
            print(f"    Resizing image from {original_width}x{original_height} to {target_width}x{target_height} for SVD.")
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"    Error resizing to 1080p: {e}. Falling back to 1024x576.")
        # Fall back to standard SVD-XT resolution
        target_width, target_height = (1024, 576)
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    frames = i2v_pipe(
        image,
        decode_chunk_size=decode_chunk_size, # Lower if OOM, higher for speed if VRAM allows
        motion_bucket_id=motion_bucket_id,
        num_frames=num_frames,
        fps=fps,
        noise_aug_strength=noise_aug_strength # Small amount of noise can improve video quality
    ).frames[0]
    return frames


def process_scene(scene_data, output_dir="output"):
    scene_id = scene_data.get("id", "unknown_scene")
    scene_desc = scene_data.get("scene_desc", "No scene description provided.")
    motion_desc_log = scene_data.get("motion_desc", "No motion description.") # For logging
    
    # Get T2I parameters or use defaults
    t2i_params = scene_data.get("t2i_params", {})
    t2i_steps = t2i_params.get("num_inference_steps", 30)  # Increased from 25 for better quality
    t2i_guidance = t2i_params.get("guidance_scale", 7.5)
    t2i_width = t2i_params.get("width", 1024)  # Increased from 512
    t2i_height = t2i_params.get("height", 768)  # Increased from 512 for proper aspect ratio
    t2i_negative_prompt = t2i_params.get("negative_prompt", "blurry, low quality, unrealistic")

    # Get I2V parameters or use defaults
    i2v_params = scene_data.get("i2v_params", {})
    i2v_motion_bucket_id = i2v_params.get("motion_bucket_id", 127)
    i2v_fps = i2v_params.get("fps", 24)  # Increased from 7 for smoother video
    i2v_num_frames = i2v_params.get("num_frames", 25) # SVD-XT default is 25
    i2v_decode_chunk_size = i2v_params.get("decode_chunk_size", 8)
    i2v_noise_aug_strength = i2v_params.get("noise_aug_strength", 0.02)


    print(f"\nProcessing scene: {scene_id}")
    print(f"  Scene Description: {scene_desc}")
    print(f"  Motion Description (for logging): {motion_desc_log}")

    start_time_scene = time.time()

    # --- 1. Text-to-Image ---
    try:
        start_time_t2i = time.time()
        generated_image = generate_image_from_text(
            prompt=scene_desc,
            negative_prompt=t2i_negative_prompt,
            num_inference_steps=t2i_steps,
            guidance_scale=t2i_guidance,
            width=t2i_width,
            height=t2i_height
        )
        t2i_duration = time.time() - start_time_t2i
        print(f"  Image generated in {t2i_duration:.2f} seconds.")

        # Save intermediate image (optional, but good for debugging)
        temp_image_filename = f"{scene_id}_base_image.png"
        temp_image_path = os.path.join(output_dir, temp_image_filename)
        generated_image.save(temp_image_path)
        print(f"  Saved base image: {temp_image_path}")

    except Exception as e:
        print(f"Error during Text-to-Image generation for scene {scene_id}: {e}")
        torch.cuda.empty_cache() # Clear cache on error
        return # Skip to next scene

    # --- 2. Image-to-Video ---
    try:
        start_time_i2v = time.time()
        video_frames = generate_video_from_image(
            image=generated_image,
            motion_bucket_id=i2v_motion_bucket_id,
            fps=i2v_fps,
            num_frames=i2v_num_frames,
            decode_chunk_size=i2v_decode_chunk_size,
            noise_aug_strength=i2v_noise_aug_strength
        )
        i2v_duration = time.time() - start_time_i2v
        print(f"  Video frames generated in {i2v_duration:.2f} seconds.")

        video_filename = f"{scene_id}_video.mp4"
        video_output_path = os.path.join(output_dir, video_filename)
        export_to_video(video_frames, video_output_path, fps=i2v_fps)
        print(f"  Generated video saved: {video_output_path}")

    except Exception as e:
        print(f"Error during Image-to-Video generation for scene {scene_id}: {e}")
        # Fall through to cleanup

    finally:
        # Clear CUDA cache after each full scene processing to free VRAM
        # This is crucial if processing many scenes sequentially.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("  CUDA cache cleared for scene.")
    
    scene_duration = time.time() - start_time_scene
    print(f"Scene {scene_id} processed in {scene_duration:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iteratively process JSON data to generate images and videos.")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated media.")
    
    args = parser.parse_args()

    json_file_path = args.json_file
    output_directory = args.output_dir

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        exit(1)

    # Check if this is a scenes array or a content_mode-based JSON
    scenes_data = []
    t2i_model_id = DEFAULT_T2I_MODEL_ID
    i2v_model_id = DEFAULT_I2V_MODEL_ID
    
    if isinstance(data, list):
        # Direct list of scenes
        scenes_data = data
    elif isinstance(data, dict):
        # YouTube-creator style format with content_mode
        # Extract scenes array
        scenes_data = data.get("scene_directions", [])
        
        # Check for model settings
        if "content_mode" in data:
            # Read model ID from content mode if provided
            if "getimg_model" in data:
                t2i_model_id = data["getimg_model"]
                print(f"Using content mode model: {t2i_model_id}")
        
        # Convert scene_directions to expected format for processing
        processed_scenes = []
        for scene in scenes_data:
            # Map to the format expected by process_scene
            processed_scene = {
                "id": f"scene_{scene.get('position', 0)}",
                "scene_desc": scene.get("scene_desc", ""),
                "motion_desc": scene.get("motion_desc", ""),
                "t2i_params": {
                    "num_inference_steps": data.get("num_inference_steps", 30),
                    "guidance_scale": data.get("guidance_scale", 7.5),
                    "width": data.get("width", 1024),  # Changed from 512
                    "height": data.get("height", 768),  # Changed from 512
                    "negative_prompt": "blurry, low quality, unrealistic"
                },
                "i2v_params": {
                    "motion_bucket_id": data.get("motion_bucket_id", 127),
                    "fps": data.get("fps", 24),  # Increased from 7 for smoother video
                    "num_frames": data.get("num_frames", 25),
                    "decode_chunk_size": 8,
                    "noise_aug_strength": 0.02
                }
            }
            processed_scenes.append(processed_scene)
        
        # Update scenes_data with the processed scenes
        scenes_data = processed_scenes

    if not scenes_data:
        print("Error: No scene data found in JSON.")
        exit(1)

    print("Starting batch processing...")
    total_start_time = time.time()

    try:
        # Load models once with the appropriate model IDs
        load_models(t2i_model_id, i2v_model_id)
        
        for i, scene_entry in enumerate(scenes_data):
            print(f"--- Scene {i+1}/{len(scenes_data)} ---")
            process_scene(scene_entry, output_dir=output_directory)
    except Exception as e:
        print(f"A critical error occurred during model loading or batch processing: {e}")
    finally:
        # Clean up models from GPU memory
        if 't2i_pipe' in globals() and t2i_pipe is not None:
            del globals()['t2i_pipe']
        if 'i2v_pipe' in globals() and i2v_pipe is not None:
            del globals()['i2v_pipe']
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Models unloaded and final CUDA cache cleared.")

    total_duration = time.time() - total_start_time
    print(f"\nFinished processing all scenes in {total_duration:.2f} seconds ({total_duration/60:.2f} minutes).")
