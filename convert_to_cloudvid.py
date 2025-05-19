#!/usr/bin/env python3
"""
CloudVid YouTube Format Converter

This script converts JSON files from the YouTube Video Creator format 
to CloudVid's compatible format.
"""

import json
import argparse
import os
import time
from datetime import datetime

def extract_aspect_ratio(scene_desc):
    """
    Extract aspect ratio notation (::number) from scene description.
    Returns the aspect ratio as a float if found, otherwise None.
    """
    if not scene_desc or not isinstance(scene_desc, str):
        return None
        
    import re
    # Look for ::n.n format (e.g., ::1.3)
    match = re.search(r'::([\d\.]+)', scene_desc)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None

def convert_scene(scene, index):
    """
    Convert a YouTube Video Creator scene to CloudVid format.
    
    Args:
        scene: Dictionary containing YouTube scene data
        index: Scene index for ID generation
        
    Returns:
        Dictionary in CloudVid format
    """
    # Get scene description and motion description
    scene_desc = scene.get('scene_desc', '')
    motion_desc = scene.get('motion_desc', '')
    
    # Extract aspect ratio if present in scene_desc
    aspect_ratio = extract_aspect_ratio(scene_desc)
    
    # Determine appropriate dimensions based on aspect ratio
    width, height = 1024, 768  # Default to 1024x768
    if aspect_ratio:
        if aspect_ratio > 1:  # Landscape
            width, height = 1024, int(1024 / aspect_ratio)
        else:  # Portrait or square
            height, width = 768, int(768 * aspect_ratio)
    
    # Handle scenes with events (sub-scenes)
    if 'events' in scene and isinstance(scene['events'], list) and scene['events']:
        # For scenes with events, we'll create a main scene with the first event
        # and additional scenes for the other events
        cloudvid_scenes = []
        
        # Create main scene from the first event
        first_event = scene['events'][0]
        main_scene = {
            "id": f"scene_{index}",
            "scene_desc": first_event.get('scene_desc', scene_desc),
            "motion_desc": first_event.get('motion_desc', motion_desc),
            "t2i_params": {
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "width": width,
                "height": height,
                "negative_prompt": "blurry, low quality, watermark, text, deformed"
            },
            "i2v_params": {
                "motion_bucket_id": 127,
                "fps": 24,
                "num_frames": 25,
                "decode_chunk_size": 8,
                "noise_aug_strength": 0.02
            },
            "metadata": {
                "original_data": scene,
                "is_event": True,
                "event_index": 0
            }
        }
        cloudvid_scenes.append(main_scene)
        
        # Create scenes for additional events
        for event_idx, event in enumerate(scene['events'][1:], 1):
            event_scene = {
                "id": f"scene_{index}_event_{event_idx}",
                "scene_desc": event.get('scene_desc', ''),
                "motion_desc": event.get('motion_desc', ''),
                "t2i_params": {
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5,
                    "width": width,
                    "height": height,
                    "negative_prompt": "blurry, low quality, watermark, text, deformed"
                },
                "i2v_params": {
                    "motion_bucket_id": 127,
                    "fps": 24,
                    "num_frames": 25,
                    "decode_chunk_size": 8,
                    "noise_aug_strength": 0.02
                },
                "metadata": {
                    "original_event": event,
                    "parent_scene": scene,
                    "is_event": True,
                    "event_index": event_idx
                }
            }
            cloudvid_scenes.append(event_scene)
            
        return cloudvid_scenes
    else:
        # Regular scene without events
        return [{
            "id": f"scene_{index}",
            "scene_desc": scene_desc,
            "motion_desc": motion_desc,
            "t2i_params": {
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "width": width,
                "height": height,
                "negative_prompt": "blurry, low quality, watermark, text, deformed"
            },
            "i2v_params": {
                "motion_bucket_id": 127,
                "fps": 24,
                "num_frames": 25,
                "decode_chunk_size": 8,
                "noise_aug_strength": 0.02
            },
            "metadata": {
                "original_data": scene
            }
        }]

def convert_youtube_json(input_path, output_path=None, preserve_original=True):
    """
    Convert a YouTube Video Creator JSON file to CloudVid format.
    
    Args:
        input_path: Path to the YouTube JSON file
        output_path: Path for the output file (optional)
        preserve_original: Whether to create an enriched version with original data
        
    Returns:
        Tuple of (output_path, enriched_path)
    """
    try:
        # Load the YouTube format JSON
        with open(input_path, 'r', encoding='utf-8') as f:
            youtube_data = json.load(f)
        
        # Check if it has the expected structure
        if not isinstance(youtube_data, dict) or 'scene_directions' not in youtube_data:
            raise ValueError("Input JSON does not appear to be in YouTube Video Creator format")
            
        scenes = youtube_data.get('scene_directions', [])
        if not scenes or not isinstance(scenes, list):
            raise ValueError("No scene directions found in the input JSON")
            
        # Convert scenes to CloudVid format
        cloudvid_scenes = []
        for i, scene in enumerate(scenes):
            converted_scenes = convert_scene(scene, i)
            cloudvid_scenes.extend(converted_scenes)
            
        # Get output paths
        if not output_path:
            # Create a timestamped output path in the same directory as input
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_dir = os.path.dirname(input_path)
            input_filename = os.path.basename(input_path)
            input_basename = os.path.splitext(input_filename)[0]
            
            output_path = os.path.join(input_dir, f"{input_basename}_cloudvid_{timestamp}.json")
        
        # Path for enriched version with original data
        enriched_path = None
        if preserve_original:
            enriched_path = output_path.replace('.json', '_with_original.json')
            
            # Create enriched version with metadata
            enriched_data = {
                "original_youtube_data": youtube_data,
                "converted_scenes": cloudvid_scenes,
                "metadata": {
                    "conversion_time": datetime.now().isoformat(),
                    "source_file": input_path,
                    "cloudvid_version": "1.0.0"
                }
            }
            
            with open(enriched_path, 'w', encoding='utf-8') as f:
                json.dump(enriched_data, f, indent=2, ensure_ascii=False)
                
        # Write the CloudVid format JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cloudvid_scenes, f, indent=2, ensure_ascii=False)
            
        return output_path, enriched_path
            
    except Exception as e:
        print(f"Error converting YouTube JSON: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Convert YouTube Video Creator JSON to CloudVid format")
    parser.add_argument("input_path", help="Path to the YouTube Video Creator JSON file")
    parser.add_argument("--output", "-o", help="Output path for CloudVid JSON file")
    parser.add_argument("--no-preserve", action="store_true", help="Don't create enriched version with original data")
    
    args = parser.parse_args()
    
    try:
        output_path, enriched_path = convert_youtube_json(
            args.input_path, 
            args.output, 
            not args.no_preserve
        )
        
        print(f"Conversion successful!")
        print(f"CloudVid JSON saved to: {output_path}")
        
        if enriched_path:
            print(f"Enriched JSON with original data saved to: {enriched_path}")
            
    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 