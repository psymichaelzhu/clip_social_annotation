"""
Extract video embeddings using pre-trained RGB ResNet model.
Modified from original Moments in Time test script https://github.com/zhoubolei/moments_models/tree/v2?tab=readme-ov-file.
main modifications: 
- The original file was based on the command line, but here it has been changed to code batch processing.
- The source file only outputs the classification category, but here a hook function is used to obtain the embedding.
"""
#%% packages
import os
import torch
import torch.nn.parallel
from torch.nn import functional as F
import numpy as np
import pandas as pd
import models
from utils import extract_frames

#%% function
def extract_video_embedding(video_file, model_type='resnet3d50', num_segments=16):
    """
    Extract embedding from a video using pre-trained model
    
    Args:
        video_file (str): Path to video file
        model_type (str): Type of model to use ('resnet50' or 'resnet3d50' or 'multi_resnet3d50')
        num_segments (int): Number of segments to extract from video
        
    Returns:
        np.ndarray: Video embedding
    """
    # Load model
    model = models.load_model(model_type)
    transform = models.load_transform()
    
    # Extract frames
    frames = extract_frames(video_file, num_segments)
    
    # Prepare input tensor
    if 'resnet3d50' in model_type:
        # [1, num_frames, 3, 224, 224]
        input_tensor = torch.stack([transform(frame) for frame in frames], 1).unsqueeze(0)
    else:
        # [num_frames, 3, 224, 224]
        input_tensor = torch.stack([transform(frame) for frame in frames])
    
    # Extract embedding from the last layer before classification
    with torch.no_grad():
        features = None
        
        # Define hook function to capture intermediate layer output
        def hook_fn(module, input, output):
            nonlocal features
            features = output.clone()
        
        # Register hook to the layer before final classification
        model.avgpool.register_forward_hook(hook_fn)
            
        # Forward pass
        _ = model(input_tensor)
        
        # Average across frames if using 2D model
        if 'resnet3d50' not in model_type:
            features = features.mean(dim=0, keepdim=True)
    
    # Convert to numpy array
    embedding = features.cpu().numpy().squeeze()
    print(embedding.shape)
    return embedding
    
def process_video_folder(video_dir, model_type='resnet3d50'):
    """
    Process all videos in a directory and return embeddings as a dataframe
    
    Args:
        video_dir (str): Directory containing videos
        output_dir (str): Directory to save embeddings
        model_type (str): Type of model to use
        
    Returns:
        pd.DataFrame: DataFrame with video names and embeddings
    """
    
    # Get list of video files
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    # Process each video
    embeddings_list = []
    for i, video_file in enumerate(video_files):
        print(f"\n {model_type} [{i+1}/{len(video_files)}] Processing {video_file}...")
        video_path = os.path.join(video_dir, video_file)
        
        try:
            embedding = extract_video_embedding(video_path, model_type)
            # Create dictionary with video name and each dimension as separate columns
            video_dict = {'video_name': video_file}
            for dim_idx, dim_val in enumerate(embedding):
                video_dict[f'dim_{dim_idx+1}'] = dim_val
            embeddings_list.append(video_dict)
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
            continue
    
    # Create DataFrame with video name and dimension columns
    df = pd.DataFrame(embeddings_list)
    
    return df

#%% main
if __name__ == "__main__":
    # Configuration
    VIDEO_DIR = '../../data/MiT_original_videos'
    MODEL_TYPES = ['resnet3d50', 'multi_resnet3d50']
    SAVE_PATH = '../../data/embedding/resnet_for_rsm.npy'
    
    # Check if the file exists and load if it does
    if os.path.exists(SAVE_PATH):
        print("Loading existing embeddings...")
        resnet_for_rsm = np.load(SAVE_PATH, allow_pickle=True)
    else:
        # Process videos with each model type
        resnet_for_rsm = {}
        for model_type in MODEL_TYPES:
            print(f"\nProcessing videos with {model_type}...")
            df = process_video_folder(VIDEO_DIR, model_type)
            resnet_for_rsm[model_type] = df
        
        # Save combined results
        np.save(SAVE_PATH, resnet_for_rsm)
    
    resnet_for_rsm

#%%
resnet_for_rsm


# %%
