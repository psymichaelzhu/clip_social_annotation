#%% Objective
# get annotation dataframe for each hypothesis dimension


#%% packages
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

import cv2
from open_clip import create_model_and_transforms, get_tokenizer 
import torch

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from sklearn.metrics.pairwise import cosine_similarity
#%% video info
video_dir = '../data/MiT_original_videos'
video_name_list = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]


#%% Load CLIP model 
# load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model_info = ('ViT-H-14-378-quickgelu', 'dfn5b')  #84.4% zero shot accuracy on ImageNet; https://github.com/mlfoundations/open_clip
#previously: ViT-bigG-14-quickgelu metaclip_fullcc

model, _, preprocess = create_model_and_transforms(model_info[0], pretrained=model_info[1])
tokenizer = get_tokenizer(model_info[0])

# %% Load embeddings
video_embedding_df = pd.read_csv('../data/embedding/CLIP_video.csv')
video_embedding_df.head()

#%% function
def extract_frame(video_name, video_dir='../data/MiT_original_videos', target_size=224):
    """
    Extract first frame from a video and process it to a square shape for display
    
    Args:
        video_name (str): Name of the video file
        video_dir (str): Directory containing the video
        target_size (int): Target size for the square frame
        
    Returns:
        frame: First frame of the video in RGB format, cropped to square and resized
    """
    # Open video file
    video_path = os.path.join(video_dir, video_name)
    cap = cv2.VideoCapture(video_path)
    
    # Read first frame
    ret, frame = cap.read()
    if ret:
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Crop to square using shorter edge
        height, width = frame_rgb.shape[:2]
        
        if height > width:
            start = (height - width) // 2
            frame_rgb = frame_rgb[start:start+width, :, :]
        else:
            start = (width - height) // 2
            frame_rgb = frame_rgb[:, start:start+height, :]
            
        # Resize to target size
        frame_rgb = cv2.resize(frame_rgb, (target_size, target_size))
    else:
        frame_rgb = None
    
    # Release video capture
    cap.release()
    return frame_rgb

def extract_dimension_embedding(annotation_dimension_dict):
    """
    Extract CLIP text embeddings for each annotation dimension, that is, embeddings for all labels in the dimension
    
    Args:
        annotation_dimension_dict(dict): Dictionary mapping dimension names to lists of labels
                                       e.g. {'relationship': ['siblings', 'couple', 'coworkers']}
        
    Returns:
        embeddings_dict(dict): Dictionary mapping dimension names to DataFrames containing CLIP embeddings for each label in that dimension
    """
    embeddings_dict = {}
    for dimension, labels in annotation_dimension_dict.items():
        # Create text inputs for this dimension's labels
        text_inputs = torch.cat([
            tokenizer(f"this is a picture of {label}") 
            for label in labels
        ]).to(device)
        
        # Extract features
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
        text_features = text_features.cpu().numpy()
        
        # Create dataframe for this dimension
        df = pd.DataFrame(text_features)
        df['label'] = labels
        
        embeddings_dict[dimension] = df
        
    return embeddings_dict

def annotate_one_dimension(dimension_embedding_df, video_embedding_df):
    """
    Calculate scores for each video to be classified into one of the categories within this dimension
    
    Args:
        dimension_embedding_df(DataFrame): DataFrame containing label embeddings for one dimension
        video_embedding_df(DataFrame): DataFrame containing video embeddings
        
    Returns:
        results_df(DataFrame): DataFrame with videos as rows and labels as columns, containing normalized similarity scores
    """
    # Create copy to avoid modifying original dataframes
    video_df = video_embedding_df.copy()
    dimension_df = dimension_embedding_df.copy()
    
    # Get embeddings
    video_features = video_df.drop(['video_name'], axis=1, errors='ignore')
    
    label_features = dimension_df.drop(['label'], axis=1, errors='ignore')
    
    # Calculate cosine similarity between each video and each instance
    similarities = cosine_similarity(
        video_features,
        label_features
    )
    
    results_df = pd.DataFrame(
        similarities,
        columns=dimension_df['label']
    )
    
    # Normalize rows to sum to 1
    if len(results_df.columns) > 1:
        results_df = results_df.div(results_df.sum(axis=1), axis=0)
    
    # Add video_name as column instead of index
    results_df.insert(0, 'video_name',video_df['video_name'])
    
    return results_df

def annotate_all_dimensions(annotation_dimension_dict, video_embedding_df):
    """
    Annotate all dimensions in the video_embedding_df using the dimension_embedding_df
    
    Args:
        annotation_dimension_dict(dict): Dictionary mapping dimension names to lists of instance names
        video_embedding_df(DataFrame): DataFrame containing video embeddings
        
    Returns:
        results(dict): Dictionary with dimension names as keys and DataFrames as values, containing normalized similarity scores
    """ 
    results = {}
    dimension_embeddings = extract_dimension_embedding(annotation_dimension_dict)
    for dimension, dimension_embedding_df in dimension_embeddings.items():
        results[dimension] = annotate_one_dimension(dimension_embedding_df, video_embedding_df)
    return results
def visualize_annotation_distributions(annotation_df, color_map=plt.cm.rainbow,video_dir='../data/MiT_original_videos'):
    """
    Visualize distributions and example frames for each instance in a dimension
    
    Args:
        annotation_df(DataFrame): DataFrame containing annotation scores for each instance, 
                                     with video_name as first column
    """
    # Get instance columns (all columns except video_name)
    categories = [col for col in annotation_df.columns if col != 'video_name']
    n_categories = len(categories)
    
    # Handle single instance case
    if n_categories == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        axes = [ax]  # Make axes iterable for consistent code below
    else:
        fig, axes = plt.subplots(n_categories, 1, figsize=(8, 4*n_categories))
        
    colors = color_map(np.linspace(0, 1, n_categories))

    for i, instance in enumerate(categories):
        ax = axes[i] if n_categories > 1 else axes[0]
        # Plot histogram in main subplot with color from colormap
        ax.hist(annotation_df[instance], bins=20, density=True, alpha=0.7, color=colors[i])
        ax.set_xlabel('Item Score', fontsize=20)
        ax.set_ylabel('Proportion', fontsize=20)
        ax.set_title(f'{instance}', fontsize=22)
        
        # Get videos at different quantiles
        quantiles = [0.02, 0.25, 0.5, 0.75, 0.98]
        quantile_scores = np.quantile(annotation_df[instance], quantiles)
        
        # Find nearest videos to each quantile
        example_videos = []
        for q_score in quantile_scores:
            nearest_idx = (annotation_df[instance] - q_score).abs().idxmin()
            nearest_video = annotation_df.loc[nearest_idx, 'video_name']
            example_videos.append(nearest_video)
        
        # Add example frames to histogram
        for video in example_videos:
            frame = extract_frame(video, target_size=100)
            score = annotation_df[annotation_df['video_name'] == video][instance].iloc[0]
            
            if frame is not None:
                imagebox = OffsetImage(frame, zoom=0.5)
                ab = AnnotationBbox(imagebox,
                                  (score, ax.get_ylim()[1]*2/3),
                                  frameon=False)
                ax.add_artist(ab)
            
            # Add vertical line in same color as histogram
            ax.axvline(x=score, color=colors[i], linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def show_extreme_videos(annotation_df, instance, video_dir='../data/MiT_original_videos'):
    """
    Show top 5 and bottom 5 videos for a given annotation dimension
    
    Args:
        annotation_df(DataFrame): DataFrame containing annotation scores for each instance, with video_name as first column
        instance(str): Column name of the instance to analyze
        video_dir(str): Directory containing the video files
    
    Returns:
        None: Displays a figure showing extreme examples
    """
    # Sort videos by the dimension score
    sorted_df = annotation_df.sort_values(by=instance)
    
    # Get top and bottom 5 videos
    bottom_5 = sorted_df.head(5)
    top_5 = sorted_df.tail(5)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    plt.rcParams.update({'font.size': 22})  # Increased font size
    
    # Plot top 5  
    for idx in range(5):
        ax = axes[0, idx]
        video = top_5.iloc[idx]['video_name'] 
        score = top_5.iloc[idx][instance]
        
        frame = extract_frame(video, target_size=224)
        if frame is not None:
            ax.imshow(frame)
            ax.set_title(f'Score: {score:.2f}', fontsize=22)  # Increased title font size
        ax.axis('off')

    
    # Plot bottom 5
    for idx in range(5):
        ax = axes[1, idx]
        video = bottom_5.iloc[idx]['video_name']
        score = bottom_5.iloc[idx][instance]
        
        frame = extract_frame(video, target_size=224)
        if frame is not None:
            ax.imshow(frame)
            ax.set_title(f'Score: {score:.2f}', fontsize=22)  # Increased title font size
        ax.axis('off')
    
    
    plt.suptitle(f'{instance}', fontsize=30)  # Increased suptitle font size
    plt.tight_layout()
    plt.show()

def visualize_dimension_embeddings(dimension_dict):
    """
    Visualize embeddings of dimension names and their categories in UMAP space
    
    Args:
        dimension_dict (dict): Dictionary mapping dimension names to lists of instance names
        
    Returns:
        None: Displays a UMAP plot
    """
    import umap
    
    # Get all words and their types (dimension vs instance)
    all_words = []
    word_types = []  # 'dimension' or 'instance' 
    dimension_ids = []  # which dimension each word belongs to
    
    for i, (dimension, categories) in enumerate(dimension_dict.items()):
        all_words.append(dimension)
        word_types.append('dimension')
        dimension_ids.append(i)
        
        for instance in categories:
            all_words.append(instance)
            word_types.append('instance')
            dimension_ids.append(i)
    
    # Get CLIP embeddings for all words
    with torch.no_grad():
        text = tokenizer(all_words).to(device)
        text_features = model.encode_text(text)
        text_features = text_features.cpu().numpy()
    
    # UMAP dimensionality reduction
    reducer = umap.UMAP(random_state=42)
    embeddings_2d = reducer.fit_transform(text_features)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Dark2(np.linspace(0, 1, len(dimension_dict)))
    
    for i in range(len(all_words)):
        if word_types[i] == 'dimension':
            plt.scatter(embeddings_2d[i,0], embeddings_2d[i,1], 
                       c=[colors[dimension_ids[i]]], marker='s', s=150)  # Increased marker size
        else:
            plt.scatter(embeddings_2d[i,0], embeddings_2d[i,1], 
                       c=[colors[dimension_ids[i]]], marker='o', s=100)  # Increased marker size
            
        plt.annotate(all_words[i], 
                    (embeddings_2d[i,0], embeddings_2d[i,1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=22)  # Increased annotation font size
    
    plt.title('Embeddings for Dimension and Instance', fontsize=26)  # Increased title font size
    plt.xlabel('UMAP 1', fontsize=24)  # Increased label font size
    plt.ylabel('UMAP 2', fontsize=24)  # Increased label font size
    plt.xticks(fontsize=22)  # Increased tick font size
    plt.yticks(fontsize=22)  # Increased tick font size
    plt.tight_layout()
    plt.show()


#%% Example usage
if False:
    dimension_dict = {
        'emotion': ['positive', 'negative'],
        'relationship': ['parent-child', 'couple', 'coworkers', 'friends','neighbors'],
        'fishing': ['fishing']
    }
    annotation_results = annotate_all_dimensions(dimension_dict, video_embedding_df)
    print(annotation_results['fishing'].head())
    print(annotation_results['emotion'].head())
    # Plot extreme videos for each instance
    for dimension, results_df in annotation_results.items():
        categories = results_df.columns[1:]  # Skip video_name column
        visualize_annotation_distributions(results_df)
        for instance in categories:
            show_extreme_videos(results_df, instance, video_dir)

    # visualize dimension embeddings (text embeddings)
    visualize_dimension_embeddings(dimension_dict)

# %% formal annotation
dimension_dict = {
        'valence': ['positive', 'negative'],
        'relationship': ['parent-child', 'couple', 'coworkers', 'friends','neighbors'],
        "activity": ["people dancing","people playing sports","people playing instruments","people cooking","people fishing"],
        "communication_modality": ["Verbal communication", "Non-verbal communication"],
        "demographics": ["male", "female","child"],
        "context": ["indoor", "yard", "wild"],
        "joint_action": ["joint action","independent action"],
        "transitivity": ["people interacting with objects","people interacting with other people","people acting independently"],
        "engagement": ["people showing high engagement","people showing low engagement"],
        "game theory": ["cooperation","competition","coordination"],
        "face orientation": ["face looking directly at the camera", "face looking to the left", "face looking to the right", "face looking down", "face looking up"],
        "face visibility": ["face fully visible", "face partially visible", "face not visible", "multiple faces visible"],
        "face distance": ["face in close-up", "face at medium distance", "face far away"]
    }

annotation_results = annotate_all_dimensions(dimension_dict, video_embedding_df)
np.save('../data/embedding/CLIP_for_rsm.npy', annotation_results)
# %%
target_dimension = 'activity'
visualize_annotation_distributions(annotation_results[target_dimension])
for instance in dimension_dict[target_dimension]:
    show_extreme_videos(annotation_results[target_dimension], instance)


# %%
target_dimension = 'transitivity'
visualize_annotation_distributions(annotation_results[target_dimension])
for instance in dimension_dict[target_dimension]:
    show_extreme_videos(annotation_results[target_dimension], instance)

#%%
target_dimension = 'context'
visualize_annotation_distributions(annotation_results[target_dimension])
for instance in dimension_dict[target_dimension]:
    show_extreme_videos(annotation_results[target_dimension], instance)
# %%
target_dimension = 'face orientation'
visualize_annotation_distributions(annotation_results[target_dimension])
for instance in dimension_dict[target_dimension]:
    show_extreme_videos(annotation_results[target_dimension], instance)
# %%
target_dimension = 'face visibility'
visualize_annotation_distributions(annotation_results[target_dimension])
for instance in dimension_dict[target_dimension]:
    show_extreme_videos(annotation_results[target_dimension], instance)
# %%
target_dimension = 'face distance'
visualize_annotation_distributions(annotation_results[target_dimension])
for instance in dimension_dict[target_dimension]:
    show_extreme_videos(annotation_results[target_dimension], instance)
# %%
