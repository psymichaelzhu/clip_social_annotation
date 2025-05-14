#%% Objective
# get clip embeddings for each video

#%% packages
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

import cv2
from open_clip import create_model_and_transforms, get_tokenizer 
import torch

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm

#%% video info
video_dir = '../data/MiT_original_videos'
video_name_list = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]


#%% Load CLIP model 
# load model and tokenizer
model_info = ('ViT-H-14-378-quickgelu', 'dfn5b')  #84.4% zero shot accuracy on ImageNet; https://github.com/mlfoundations/open_clip
#previously: ViT-bigG-14-quickgelu metaclip_fullcc

model, _, preprocess = create_model_and_transforms(model_info[0], pretrained=model_info[1])
tokenizer = get_tokenizer(model_info[0])


#%% functions
def extract_frame_embedding(frame):
    """
    Extract embedding vector from one frame using CLIP model
    
    Args:
        frame: Input frame (array vector)
        
    Returns:
        embedding: Embedding vector for this frame
    """
    frame_pil = Image.fromarray(frame)#PIL image object
    frame_preprocessed = preprocess(frame_pil).unsqueeze(0)
    with torch.no_grad(), torch.autocast("cuda"):
        frame_features = model.encode_image(frame_preprocessed)
    return frame_features

def extract_video_embedding(filename, num_frames=None, video_dir='../data/MiT_original_videos'):
    """
    Extract embeddings of evenly spaced frames in a video
    
    Args:
        filename: Video filename
        num_frames: Number of frames to extract
        video_dir: Directory containing videos
        
    Returns:
        video_embedding: Array of frame embeddings (frames x embedding_dim)
    """
    # Open video file
    video_path = os.path.join(video_dir, filename)
    cap = cv2.VideoCapture(video_path)

    # Calculate frame indices
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames is None: #extract all frames
        frame_indices = list(range(total_frames))
    else:
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    # Extract frames and embeddings
    frames_embeddings = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)#set frame_idx as the frame to be decoded/captured next
        ret, frame = cap.read()#output: ret: True/False; frame: an image array vector 
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# Convert BGR (OpenCV default) to RGB (model expected)
            frame_embedding = extract_frame_embedding(frame_rgb)
            frames_embeddings.append(frame_embedding)
    
    cap.release()
    return np.array(frames_embeddings)

def similarity_video_embedding(all_video_embedding):
    """
    Calculate within and between video similarities
    
    Args:
        all_video_embedding: Dictionary of video embeddings
        
    Returns:
        within_df: DataFrame of within-video similarities
        between_df: DataFrame of between-video similarities
    """
    # Calculate centroids
    centroids = {video: np.mean(emb, axis=0) for video, emb in all_video_embedding.items()}
    
    # Calculate within-video similarities
    within_data = []
    for video, embeddings in all_video_embedding.items():
        centroid = centroids[video].reshape(1, -1)#not equivalent to reshape(-1); the former is a (1, n) 2D array [[1 2 3]], the latter is a 1D array (n,) [1 2 3]
        frame_similarities = cosine_similarity(embeddings, centroid).squeeze()
        for frame_id, similarity in enumerate(frame_similarities):
            within_data.append({
                'video_name': video,
                'index': frame_id,
                'value': similarity
            })
    
    # Calculate between-video similarities
    between_data = []
    centroid_matrix = np.array(list(centroids.values()))
    centroid_similarities = cosine_similarity(centroid_matrix)
    video_names = list(centroids.keys())
    
    for i, video1 in enumerate(video_names):
        for j, video2 in enumerate(video_names):
            if i != j:
                between_data.append({
                    'video_name': video1,
                    'index': video2, 
                    'value': centroid_similarities[i,j]
                })
    
    within_df = pd.DataFrame(within_data)
    between_df = pd.DataFrame(between_data)
    
    return within_df, between_df

def visualize_video_similarity(display_video_names, within_similarities, between_similarities, label='Index', display_image=False):
    """
    Visualize within and between video similarities
    
    Args:
        display_video_names: List of video names to display
        within_similarities: DataFrame of within-video similarities
        between_similarities: DataFrame of between-video similarities
        label: Label to display on x-axis ('Name' or 'Index')
        display_image: Whether to display video frames
        
    Returns:
        None
    """
    # Filter within similarities for selected videos
    within_filtered = within_similarities[within_similarities['video_name'].isin(display_video_names)].copy()
    within_filtered['type'] = 'within'

    # Filter between similarities for selected videos
    between_filtered = between_similarities[between_similarities['video_name'].isin(display_video_names)].copy()
    between_filtered['type'] = 'between'


    # Combine dataframes
    plot_data = pd.concat([
        within_filtered, between_filtered
    ])

    # Get video indices
    video_indices = {name: idx for idx, name in enumerate(display_video_names)}
    plot_data['video_index'] = plot_data['video_name'].map(video_indices)

    # Visualize: Scatter plot with jitter
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 20}) 
    
    # Add jitter to x-axis
    jitter = 0.2
    for stype, color in zip(['within', 'between'], ['#1f77b4', '#ff7f0e']):
        mask = plot_data['type'] == stype
        x = plot_data.loc[mask, 'video_index'] + np.random.uniform(-jitter, jitter, size=mask.sum())
        plt.scatter(x, plot_data.loc[mask, 'value'], 
                   alpha=0.3, label=stype, color=color, s=40) 

    plt.xlabel(f'Video {label} (randomly picked)', fontsize=20)
    plt.ylabel('Cosine similarity', fontsize=20)
    plt.title('Within and Between Video Similarities', fontsize=22)
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right', fontsize=20, markerscale=2) 
    
    # Set x-axis labels
    if label == 'Index':
        labels = [str(video_name_list.index(name)) for name in display_video_names]
        plt.xticks(range(len(display_video_names)), labels, fontsize=20)
    else: 
        plt.xticks(range(len(display_video_names)), display_video_names, rotation=45, fontsize=20)
        
    # Display first frame of each video above x-axis if display_image is True
    if display_image:
        for i, name in enumerate(display_video_names):
            cap = cv2.VideoCapture(os.path.join(video_dir, name))
            ret, frame = cap.read()
            if ret:
                ax = plt.axes([0.1 + i*0.8/len(display_video_names), 0.7, 0.8/len(display_video_names), 0.2])
                ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ax.axis('off')
            cap.release()
    
    plt.tight_layout()
    plt.show()

def visualize_dimension_distribution(df, n_dim=5):
    """
    Visualize distribution of randomly selected CLIP dimensions using violin plots
    
    Args:
        df: DataFrame containing CLIP embeddings
        n_dim: Number of dimensions to visualize
        random_state: Random seed for reproducibility
    """
    # Get CLIP feature columns
    clip_cols = [col for col in df.columns if col.startswith('CLIP_')]
    
    # Randomly select dimensions
    np.random.seed(42)
    selected_dims = np.random.choice(clip_cols, size=n_dim, replace=False)
    
    # Create violin plot
    plt.figure(figsize=(12, 6))
    plt.violinplot([df[dim] for dim in selected_dims])
    
    # Extract just the numbers from the dimension names
    dim_numbers = [dim.split('_')[1] for dim in selected_dims]
    plt.xticks(range(1, n_dim+1), dim_numbers)
    plt.xlabel('CLIP Dimension Index')
    plt.ylabel('Values')
    plt.title('Distribution of Randomly Picked Dimensions')
    
    plt.tight_layout()
    plt.show()
#%% main: 1. Extract embeddings for all videos
file_dir = '../data/embedding'
file_name = 'CLIP_video.npy'
file_path = os.path.join(file_dir, file_name)
if not os.path.exists(file_path):
    all_video_embedding = {}
    for video_file in tqdm(video_name_list, desc="Processing videos"):
        video_embedding = extract_video_embedding(video_file, num_frames=None)
        all_video_embedding[video_file] = video_embedding.squeeze()

    # Save all video embeddings to npy file
    if not os.path.exists(file_path):
        os.makedirs(file_dir, exist_ok=True)

    np.save(file_path, all_video_embedding)
    print(f"Saved video embeddings to {file_path}")
else:
    print(f'Loading video embeddings from {file_path}')
    all_video_embedding = np.load(file_path, allow_pickle=True)
    all_video_embedding = all_video_embedding.item()
    
print("number of videos:", len(all_video_embedding.keys()))
print("shape of the first video embedding \n frame x embedding dimension:", all_video_embedding[video_name_list[0]].shape)

#%% main: 2. Validate the rationale of averaging frame embeddings to get video embeddings

# Compare within-video similarity and between-video similarity
within_similarities, between_similarities = similarity_video_embedding(all_video_embedding)

random_indices = np.random.RandomState(42).choice(len(video_name_list), size=5, replace=False)
selected_videos = [video_name_list[i] for i in random_indices]
visualize_video_similarity(selected_videos, within_similarities, between_similarities, label='Index', display_image=False)
#%% main: 3. Aggregating video-level embeddings
video_embeddings_df = pd.DataFrame()
for video_name, frame_embeddings in all_video_embedding.items():
    mean_embedding = np.mean(frame_embeddings, axis=0)
    
    temp_df = pd.DataFrame([mean_embedding], columns=[f'CLIP_{i+1}' for i in range(len(mean_embedding))])
    temp_df.insert(0, 'video_name', video_name)
    
    video_embeddings_df = pd.concat([video_embeddings_df, temp_df], ignore_index=True)

print("Shape of video embeddings dataframe:", video_embeddings_df.shape)
print("\nFirst few rows:")
print(video_embeddings_df.head())

# Save video embeddings dataframe to csv file
file_name = 'CLIP_video.csv'
video_embeddings_df.to_csv(os.path.join(file_dir, file_name), index=False)
print(f"Saved video embeddings dataframe to {os.path.join(file_dir, file_name)}")

#%% main: 4. Visualize feature distribution
visualize_dimension_distribution(video_embeddings_df, n_dim=8)

