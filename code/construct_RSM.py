# %% packages
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import cosine, euclidean, jaccard
from scipy.stats import pearsonr
#%% video info
video_dir = '../data/MiT_original_videos'
video_name_list = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
len(video_name_list)
# %% function
def compute_sequence_similarity(sequence1, sequence2, similarity_metric='cosine'):
    """Compute similarity between two sequences using specified metric
    Args:
        sequence1 (array-like): First sequence
        sequence2 (array-like): Second sequence 
        similarity_metric (str): Similarity metric to use ('cosine', 'pearson', 'euclidean', or 'jaccard')
    Returns:
        float: Similarity value between the sequences
    """
    # Convert to numpy arrays
    seq1 = np.array(sequence1)
    seq2 = np.array(sequence2)
    
    # Find indices of non-NA values in both sequences
    valid_idx = ~(np.isnan(seq1) | np.isnan(seq2))
    
    # Remove NA values
    seq1_clean = seq1[valid_idx]
    seq2_clean = seq2[valid_idx]
    
    if similarity_metric == 'cosine':
        return 1-cosine(seq1_clean, seq2_clean)
    elif similarity_metric == 'pearson':
        corr, _ = pearsonr(seq1_clean, seq2_clean)  # Only take correlation coefficient
        return corr
    elif similarity_metric == 'euclidean':
        return 1 - euclidean(seq1_clean, seq2_clean)
    elif similarity_metric == 'jaccard':
        return 1 - jaccard(seq1_clean, seq2_clean)
    else:
        raise ValueError("Invalid similarity metric. Choose 'cosine', 'pearson', 'euclidean', or 'jaccard'")


def display_rsm(rsm):
    '''
    visualize the video-by-video rsm
    '''
    n_videos=rsm.shape[0]
    plt.figure(figsize=(10,8))
    # Show every 10th tick label
    tick_labels = [str(i) if i % 10 == 0 else '' for i in range(n_videos)]
    
    sns.heatmap(rsm, cmap='viridis', 
                xticklabels=tick_labels, 
                yticklabels=tick_labels)
    plt.title('Representational Similarity Matrix', fontsize=22)
    plt.xlabel('Video Index', fontsize=18)
    plt.ylabel('Video Index', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()


def compute_rsm(df, similarity_metric='pearson', video_name_list=None, if_display=True):
    """Compute representational similarity matrix from feature vectors
    Args:
        df (pd.DataFrame): DataFrame with video_name as first column and features as remaining columns
        similarity_metric (str): Similarity metric to use
        video_name_list (list): List of video names to include and their order
        if_display (bool): Whether to display the RSM heatmap
    Returns:
        np.ndarray: Pairwise similarity matrix
    """
    # Filter and sort by video_name_list if provided
    if video_name_list is not None:
        df = df[df['video_name'].isin(video_name_list)]
        df = df.set_index('video_name').loc[video_name_list].reset_index()
    
    # Extract features
    features = df.iloc[:, 1:].values
    n_videos = len(features)
    
    # Compute pairwise similarities
    rsm = np.zeros((n_videos, n_videos))
    for i in range(n_videos):
        for j in range(n_videos):
            rsm[i,j] = compute_sequence_similarity(features[i], features[j], similarity_metric)
    
    if if_display:
        display_rsm(rsm)
    return rsm

def compute_dict_rsm(dict_df, video_name_list, similarity_metric='pearson'):
    """Recursively convert nested dictionary of DataFrames to nested dictionary of RSMs
    
    Args:
        dict_df (dict): Nested dictionary where leaf nodes are DataFrames with video_name column
        video_name_list (list): List of video names to use for RSM computation
    
    Returns:
        dict: Nested dictionary with same structure as input but RSMs as leaf nodes
    """
    # Initialize output dictionary
    dict_rsm = {}
    
    # Recursively process dictionary
    for key, value in dict_df.items():
        if isinstance(value, dict):
            # If value is dictionary, recurse
            dict_rsm[key] = compute_dict_rsm(value, video_name_list)
        else:
            # If value is DataFrame, compute RSM
            dict_rsm[key] = compute_rsm(value, 
                                      similarity_metric=similarity_metric,
                                      video_name_list=video_name_list,
                                      if_display=False)
            
    return dict_rsm


def correlate_two_rsms(rsm1, rsm2, n_permutation=1000, if_display=True):
    """Compute correlation between two RSMs with permutation testing
    Args:
        rsm1 (np.ndarray): First RSM
        rsm2 (np.ndarray): Second RSM
        n_permutation (int): Number of permutations for null distribution
        if_display (bool): Whether to display null distribution plot
    Returns:
        tuple: (correlation coefficient, permutation p-value)
    """
    # Get lower triangular indices
    tril_idx = np.tril_indices_from(rsm1, k=-1)
    
    vec1 = rsm1[tril_idx]
    vec2 = rsm2[tril_idx]
    
    # Compute observed correlation
    observed_r, _ = spearmanr(vec1, vec2)
    
    # Permutation test
    null_dist = np.zeros(n_permutation)
    for i in range(n_permutation):
        # Shuffle first RSM by row
        perm_idx = np.random.permutation(len(vec1))
        null_dist[i], _ = spearmanr(vec1[perm_idx], vec2)

    # Compute two-tailed p-value
    p_value = (1 + np.sum(np.abs(null_dist) >= np.abs(observed_r))) / (1 + n_permutation)
    
    if if_display:
        plt.figure(figsize=(10,6))
        plt.hist(null_dist, bins=50, edgecolor='black')
        plt.axvline(x=observed_r, color='r', linestyle='--')
        plt.title('Null Distribution of Correlations', fontsize=22)
        plt.xlabel('Correlation Coefficient', fontsize=18)
        plt.ylabel('Frequency', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.annotate(f'r = {observed_r:.3f}\np = {p_value:.3f}', 
                    xy=(0.7, 0.8), xycoords='axes fraction', 
                    fontsize=18)
        plt.show()
        
    return observed_r, p_value




# %% test
# compute rsm and correlate rsms
if False:   
    df1=neural_for_rsm['01']['EVC']['l']
    df2=neural_for_rsm['02']['TPJ']['l']

    rsm1=compute_rsm(df1, similarity_metric='pearson', if_display=True)
    rsm2=compute_rsm(df2, similarity_metric='pearson', if_display=True)

    correlate_two_rsms(rsm1, rsm2, n_permutation=1000, if_display=True)

# compute dict rsm
if False:
    dict_rsm=compute_dict_rsm(neural_for_rsm['01'], video_name_list=None, similarity_metric='pearson')

    print(dict_rsm)
   
    display_rsm(dict_rsm['EVC']['l'])
   
    correlate_two_rsms(dict_rsm['EVC']['l'], dict_rsm['TPJ']['l'], n_permutation=1000, if_display=True)
# %% load neural data
neural_for_rsm=np.load('../data/neural/neural_for_rsm.npy',allow_pickle=True).item()
neural_for_rsm
# %% compute all neural rsms
if os.path.exists('../data/RSA/neural_rsm.npy'):
    print('neural rsm already exists, loading...')
    all_neural_rsms=np.load('../data/RSA/neural_rsm.npy',allow_pickle=True).item()
else:
    # Compute RSMs for subjects 01-04
    subject_rsms = {}
    for sub in ['01', '02', '03', '04']:
        print(f'computing rsm for subject {sub}')
        subject_rsms[sub] = compute_dict_rsm(neural_for_rsm[sub], 
                                        video_name_list=video_name_list,
                                        similarity_metric='pearson')

    # group neural RSM
    group_rsm = {}
    for roi in subject_rsms['01'].keys():
        group_rsm[roi] = {}
        for side in subject_rsms['01'][roi].keys():
            # Get RSMs from all subjects for this ROI and side
            rsms = [subject_rsms[sub][roi][side] for sub in ['01', '02', '03', '04']]
            
            # Fisher Z transform, then average, then convert back to correlation
            z_rsms = [np.arctanh(rsm) for rsm in rsms]
            mean_z = np.mean(z_rsms, axis=0)
            group_rsm[roi][side] = np.tanh(mean_z)

    # Combine all neural RSMs into final dictionary
    all_neural_rsms = {
        'group': group_rsm,
        'sub01': subject_rsms['01'],
        'sub02': subject_rsms['02'], 
        'sub03': subject_rsms['03'],
        'sub04': subject_rsms['04']
    }

    # Save all neural RSMs
    if not os.path.exists('../data/RSA'):
        os.makedirs('../data/RSA')
    np.save('../data/RSA/neural_rsm.npy', all_neural_rsms)
all_neural_rsms
#%% model embedding rsm
#load model embedding
#clip
clip_embedding=pd.read_csv('../data/embedding/CLIP_video.csv')
print(clip_embedding.head())

resnet_embedding_dict=np.load('../data/embedding/resnet_for_rsm.npy',allow_pickle=True).item()
model_for_rsm = {**resnet_embedding_dict, **{"CLIP":clip_embedding}}
print(model_for_rsm.keys())

# save model_for_rsm
if not os.path.exists('../data/embedding'):
    os.makedirs('../data/embedding')
np.save('../data/embedding/model_for_rsm.npy', model_for_rsm)

# RSM for model
model_rsm={}
model_rsm=compute_dict_rsm(model_for_rsm, video_name_list=video_name_list, similarity_metric='euclidean')

# save model_rsm
if not os.path.exists('../data/RSA'):
    os.makedirs('../data/RSA')
np.save('../data/RSA/model_rsm.npy', model_rsm)


#%% CLIP annotation rsm
clip_for_rsm=np.load('../data/embedding/CLIP_for_rsm.npy',allow_pickle=True).item()
clip_for_rsm

# compute CLIP RSM
clip_rsm=compute_dict_rsm(clip_for_rsm, video_name_list=video_name_list, similarity_metric='euclidean')

# save CLIP RSM
if not os.path.exists('../data/RSA'):
    os.makedirs('../data/RSA')
np.save('../data/RSA/clip_rsm.npy', clip_rsm)
print(clip_rsm.keys())




# %%
