#%% packages
from scipy.stats import spearmanr 
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.multitest import multipletests
from CLIP_dimension_annotate import annotate_all_dimensions, visualize_annotation_distributions, show_extreme_videos
from construct_RSM import compute_sequence_similarity, compute_rsm, compute_dict_rsm

#%% function

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

def correlate_dicts_asymmetric(reference_dict, candidate_dict_dict, n_permutation=1000, multiple_comparison='fdr_bh'):
    """Compute correlations between all pairs of RSMs from two dictionaries
    Args:
        reference_dict (dict): First dictionary with RSM names as keys and RSMs as values
        candidate_dict_dict (dict): Dictionary of dictionaries, where each inner dictionary contains RSM names as keys and RSMs as values
        n_permutation (int): Number of permutations for null distribution
        multiple_comparison (str): Multiple comparison correction method ('fdr_bh' or 'bonferroni')
    Returns:
        pd.DataFrame: DataFrame with correlation results including adjusted p-values and significance
    """
    # Initialize lists to store results
    results = []
    
    # Iterate through each module in candidate_dict_dict
    for module, candidate_dict in candidate_dict_dict.items():
        # Compute correlations for all pairs within this module
        for key1, rsm1 in reference_dict.items():
            for key2, rsm2 in candidate_dict.items():
                #print(f"{key1} - {module}_{key2}")
                r, p = correlate_two_rsms(rsm1, rsm2, n_permutation=n_permutation, if_display=False)
                results.append({
                    'reference': key1,
                    'candidate': key2,
                    'module': module,
                    'r': r,
                    'p': p
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Apply multiple comparison correction
    _, p_adj, _, _ = multipletests(df['p'], method=multiple_comparison)
    df['q'] = p_adj
    
    # Add significance symbols
    def get_sig_symbol(p):
        if p > 0.05:
            return 'n.s.'
        elif p > 0.01:
            return '*'
        elif p > 0.001:
            return '**'
        else:
            return '***'
    
    df['sig_sign'] = df['q'].apply(get_sig_symbol)
    
    return df

def plot_heatmap_asymmetric(correlation_df, candidate_name="Candidate RSM", reference_name="Neural RSM (by ROI)", figsize=(10,10),if_annotate=False):
    """Plot correlation matrix with significance markers
    Args:
        correlation_df (pd.DataFrame): DataFrame from correlate_two_dicts
        figsize (tuple): Figure size
    """
    def pivot_correlation_df(correlation_df, value_col):
        """Create pivot tables for correlation values and significance markers"""
        sorted_df = correlation_df.sort_values('module')
        
        ordered_refs = np.sort(sorted_df['reference'].unique())
        ordered_cands = sorted_df['candidate'].unique()
        
        matrix = sorted_df.pivot(
            index='reference',
            columns='candidate',
            values=value_col
        )
        
        matrix = matrix.reindex(columns=ordered_cands)
        matrix = matrix.reindex(index=ordered_refs)
        
        return matrix
    
    matrix = pivot_correlation_df(correlation_df, 'r')
    sig_matrix = pivot_correlation_df(correlation_df, 'sig_sign')
    
    plt.figure(figsize=figsize)
    scale_length=0.5
    
    # Get module boundaries
    modules = correlation_df['module'].unique()
    module_boundaries = []
    current_pos = 0
    for module in modules:
        module_cols = correlation_df[correlation_df['module'] == module]['candidate'].nunique()
        current_pos += module_cols
        if current_pos < len(matrix.columns):  
            module_boundaries.append(current_pos)
    
    # Plot heatmap
    g = sns.heatmap(matrix, cmap='RdBu_r', center=0, vmin=-scale_length, vmax=scale_length,
                    annot=if_annotate, fmt='.2f', cbar_kws={'label': 'Correlation'},
                    xticklabels=True, yticklabels=True)  
    
    g.set_xticklabels(g.get_xticklabels(), rotation=90, ha='center')
    g.set_yticklabels(g.get_yticklabels(), rotation=0, ha='right')
    
    ax = plt.gca()
    ax.set_xticks(np.arange(len(matrix.columns)) + 0.5)
    ax.set_yticks(np.arange(len(matrix.index)) + 0.5)
    
    # Customize colorbar
    cbar = g.collections[0].colorbar
    cbar.set_label('Spearman Correlation', fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    
    # Add white lines between modules
    for boundary in module_boundaries:
        plt.axvline(x=boundary, color='white', linewidth=2)
    
    # Add significance markers
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            if sig_matrix.iloc[i,j] != 'n.s.':
                plt.text(j+0.5, i+0.3, sig_matrix.iloc[i,j],
                        ha='center', va='center', color='black',
                        fontsize=12)
    
    plt.xlabel(candidate_name, fontsize=22)
    plt.ylabel(reference_name, fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    plt.tight_layout()
    plt.show()

def correlate_self(dict_dict, n_permutation=1000, multiple_comparison='fdr_bh', if_display=True, figsize=(10,10), scale_length=0.5):
    """
    Compute correlations between all pairs of RSMs within a dictionary of dictionaries
    
    Args:
        dict_dict (dict): Dictionary of dictionaries containing RSMs
        n_permutation (int): Number of permutations for null distribution
        multiple_comparison (str): Multiple comparison correction method
        if_display (bool): Whether to display correlation heatmap
        figsize (tuple): Figure size for heatmap
        
    Returns:
        tuple: (correlation DataFrame, correlation matrix ordered by modules)
    """
    # Compute all correlations
    results = []
    for module1, dict1 in dict_dict.items():
        for module2, dict2 in dict_dict.items():
            for key1, rsm1 in dict1.items():
                for key2, rsm2 in dict2.items():
                    r, p = correlate_two_rsms(rsm1, rsm2, 
                                            n_permutation=n_permutation, 
                                            if_display=False)
                    results.append({
                        'module': module1,
                        'reference': f"{module1}~{key1}",
                        'candidate': f"{module2}~{key2}",
                        'r': r,
                        'p': p
                    })
    
   
    df = pd.DataFrame(results)
    #  multiple comparison correction
    _, p_adj, _, _ = multipletests(df['p'], method=multiple_comparison)
    df['q'] = p_adj
    
    # Add significance symbols
    df['sig'] = pd.cut(df['q'], 
                       bins=[-np.inf, 0.001, 0.01, 0.05, np.inf],
                       labels=['***', '**', '*', 'n.s.'])
    def create_ordered_matrix(df, value_col):
        """
        Create an ordered correlation matrix from a DataFrame with module-based indices
        """
        # Get module-ordered list of all RSMs
        ordered_rsms = []
        for module in dict_dict.keys():
            module_rsms = [f"{module}~{key}" for key in dict_dict[module].keys()]
            ordered_rsms.extend(sorted(module_rsms))
            
        # Create and reorder matrix
        matrix = df.pivot(index='reference', columns='candidate', values=value_col)
        matrix = matrix.reindex(index=ordered_rsms, columns=ordered_rsms)
        
        # Remove module names from index/columns
        new_index = [x.split('~', 1)[1] for x in matrix.index]
        new_columns = [x.split('~', 1)[1] for x in matrix.columns]
        matrix.index = new_index
        matrix.columns = new_columns
        
        return matrix
        
    # Create ordered matrices
    matrix = create_ordered_matrix(df, 'r')
    sig_matrix = create_ordered_matrix(df, 'sig')
    
    if if_display:
        plt.figure(figsize=figsize)
        
        # Plot heatmap
        g = sns.heatmap(matrix, cmap='RdBu_r', center=0, vmin=-scale_length, vmax=scale_length,
                   annot=False, fmt='.2f', cbar_kws={'label': 'Correlation'})
        
        # Customize colorbar
        cbar = g.collections[0].colorbar
        cbar.set_label('Spearman Correlation', fontsize=20)
        cbar.ax.tick_params(labelsize=20)
  
        
        # Add module boundaries
        current_pos = 0
        for module in dict_dict.keys():
            current_pos += len(dict_dict[module])
            if current_pos < len(matrix):
                plt.axhline(y=current_pos, color='white', linewidth=2)
                plt.axvline(x=current_pos, color='white', linewidth=2)
        
        # Add significance markers
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if sig_matrix.iloc[i,j] != 'n.s.':
                    plt.text(j+0.5, i+0.5, sig_matrix.iloc[i,j],
                            ha='center', va='bottom', color='black',
                            fontsize=8)
        
        #plt.title('RSM Correlations', fontsize=24)
        plt.xlabel('RSMs', fontsize=22)
        plt.ylabel('RSMs', fontsize=22)
        plt.xticks(rotation=90, ha='right', fontsize=20)
        plt.yticks(rotation=0, fontsize=20)
        plt.tight_layout()
        plt.show()
    
    return df, matrix


#%% load neural rsm
neural_rsm=np.load('../data/RSA/neural_rsm.npy',allow_pickle=True).item()
# Try to load roi_neural_rsm from file, if it exists
try:
    roi_neural_rsm = np.load('../data/RSA/roi_neural_rsm.npy', allow_pickle=True).item()
except FileNotFoundError:
    # If file doesn't exist, reorganize neural RSM to combine roi and side
    roi_neural_rsm = {}

    # Process each subject/group level 
    for level in neural_rsm.keys():
        roi_neural_rsm[level] = {}
        
        # For each ROI
        for roi in neural_rsm[level].keys():
            # For each side
            for side in neural_rsm[level][roi].keys():
                # Create new key combining side and ROI
                new_key = f'{roi}_{side}'
                roi_neural_rsm[level][new_key] = neural_rsm[level][roi][side]
    
    # Save the reorganized RSM
    np.save('../data/RSA/roi_neural_rsm.npy', roi_neural_rsm)

_,_ = correlate_self({"neural":roi_neural_rsm['sub01']}, n_permutation=0, multiple_comparison='fdr_bh', if_display=True, figsize=(14,13))

#%% load model and clip rsm
# model rsm
model_rsm=np.load('../data/RSA/model_rsm.npy',allow_pickle=True).item()
print(model_rsm.keys())
# CLIP RSM
clip_rsm=np.load('../data/RSA/clip_rsm.npy',allow_pickle=True).item()
print(clip_rsm.keys())
# Combine all RSMs into one dictionary
combined_rsm = {"CLIP_annotation":clip_rsm, "model_embedding":model_rsm}
_,_ = correlate_self(combined_rsm, n_permutation=0, multiple_comparison='fdr_bh', if_display=True, figsize=(14,13))
#%% correlate candidate RSM with reorganized neural RSM for each subject/group
# Correlate combined RSM with neural RSM for each subject/group
corr_dfs = {}
for level in roi_neural_rsm.keys():
    print(level)
    print(f'\nCorrelating {level} with combined RSM:')
    corr_df = correlate_dicts_asymmetric(roi_neural_rsm[level], combined_rsm,
                                 n_permutation=0,
                                 multiple_comparison='fdr_bh') 
    plot_heatmap_asymmetric(corr_df)
    corr_dfs[level] = corr_df

# Average correlation values across subjects (excluding group)
subject_dfs = [corr_dfs[level] for level in corr_dfs.keys() if level != 'group']
average_df = subject_dfs[0].copy()

# Calculate mean correlation values
for i in range(len(average_df)):
    r_values = [df.iloc[i]['r'] for df in subject_dfs]
    average_df.iloc[i, average_df.columns.get_loc('r')] = np.mean(r_values)
    
    # Use majority vote for significance
    sig_signs = [df.iloc[i]['sig_sign'] for df in subject_dfs]
    most_common_sign = max(set(sig_signs), key=sig_signs.count)
    average_df.iloc[i, average_df.columns.get_loc('sig_sign')] = most_common_sign

print('\nAverage correlation across subjects:')
plot_heatmap_asymmetric(average_df)
#for each df (subject or average): (20 CLIP annotations + 4 model embeddings) * (11 RoIs * 2 sides) = 528








# %%
import openai
from os import environ
openai.api_key = environ.get('OPENAI_API_KEY')
#%% semantic inference
def semantic_inference(roi, r_threshold=None):
    """Perform semantic inference based on RSA correlations to predict related concepts using GPT
    
    Args:
        roi (str): Region of interest to analyze
        r_threshold (float): Correlation threshold for positive/negative examples
        
    Returns:
        list: Predicted related words from GPT model that are novel and at same semantic level
    """
    # Extract correlations for the specified ROI
    roi_corrs = average_df[(average_df['reference'] == roi) & (average_df['module'] == 'CLIP_annotation')]
    
    # Sort by correlation value
    sorted_corrs = roi_corrs.sort_values('r', ascending=False)
    
    # Split into positive and negative examples based on threshold
    if r_threshold is None:
        # Split the sorted correlations into two equal halves
        midpoint = len(sorted_corrs) // 2
        positives = sorted_corrs['candidate'].iloc[:midpoint].tolist()
        negatives = sorted_corrs['candidate'].iloc[midpoint:].tolist()
    else:
        positives = sorted_corrs[sorted_corrs['r'] >= r_threshold]['candidate'].tolist()
        negatives = sorted_corrs[sorted_corrs['r'] < r_threshold]['candidate'].tolist()
    
    print(f"\nPositive examples: {positives}")
    print(f"Negative examples: {negatives}")

    # Get all existing words
    existing_words = positives + negatives

    # Construct prompt for GPT
    prompt = f"""Given these two lists of words:
    Positive examples: {', '.join(positives)}
    Negative examples: {', '.join(negatives)}
    
    Please predict 10 new words that:
    1. [! the most important] Are semantically similar to the positive examples but dissimilar to the negative examples
    2. Are NOT in either the positive or negative examples (or their synonyms) or "interaction"
    3. Are at the same semantic level/granularity as the example words, about a potential dimension of social interaction.
    
    Return only the words as a comma-separated list, without explanations."""

    # Get GPT predictions
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for semantic prediction tasks."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    # Parse response and filter out any existing words
    predicted_words = [word.strip() for word in response.choices[0].message.content.split(',') 
                      if word.strip() not in existing_words][:10]
    
    print(f"\nNext predictions for {roi}:")
    for word in predicted_words:
        print(word)
    
    return predicted_words

#%% 0. video info
video_dir = '../data/MiT_original_videos'
video_name_list = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
len(video_name_list)

# %% 1. initial sample

# Load embeddings
video_embedding_df = pd.read_csv('../data/embedding/CLIP_video.csv')
video_embedding_df.head()

# LLM prompt engineering
dimension_dict = {
        'relationship': ['a parent and a child', 'a couple', 'two coworkers', 'two friends','two neighbors'],
        "activity": ["people dancing","people playing sports","people playing instruments","people cooking","people fishing","people doing childcare"],
        "context": ["indoor", "yard", "wild"],
        "demographics": ["male", "female","kid"],
        "emotion_binary": ["positive", "negative"],
        "emotion_multi": ["joy", "sadness", "anger", "fear", "disgust", "neutral"],
        "joint_action": ["people performing joint action","people taking independent action"],
        "Transitivity": ["people interacting with objects","people interacting with other people","people acting indepedently"],
        "communication_modality": ["People engaging in verbal communication", "People engaging in non-verbal communication","People not communicating"],
        "body_orientation": ["people direct facing each other", "people angled postures towards each other", "people turned away from each other"],
        "social_distance": ["people with close social distance", "people with far social distance"],
        "game_theory": ["coordination","competition","cooperation"], 
        "social_norm": ["people following social norm","people not following social norm"],
    }

annotation_results = annotate_all_dimensions(dimension_dict, video_embedding_df)
print(annotation_results['body_orientation'].head())
np.save('../data/embedding/CLIP_for_rsm.npy', annotation_results)
# visualize example annotation distributions
target_dimension = 'game_theory'
visualize_annotation_distributions(annotation_results[target_dimension])
for instance in dimension_dict[target_dimension]:
    show_extreme_videos(annotation_results[target_dimension], instance)
# %% 2. get performance
# 2-1 get CLIP annotation rsm
clip_for_rsm=np.load('../data/embedding/CLIP_for_rsm.npy',allow_pickle=True).item()

# compute CLIP RSM
clip_rsm=compute_dict_rsm(clip_for_rsm, video_name_list=video_name_list, similarity_metric='euclidean')

# save CLIP RSM
if not os.path.exists('../data/RSA'):
    os.makedirs('../data/RSA')
np.save('../data/RSA/clip_rsm.npy', clip_rsm)
print(clip_rsm.keys())

#%% 2-2 correlate candidate RSM with reorganized neural RSM for each subject/group

# load model and clip rsm
# model rsm
model_rsm=np.load('../data/RSA/model_rsm.npy',allow_pickle=True).item()
print(model_rsm.keys())
# CLIP RSM
clip_rsm=np.load('../data/RSA/clip_rsm.npy',allow_pickle=True).item()
print(clip_rsm.keys())
# Combine all RSMs into one dictionary
combined_rsm = {"CLIP_annotation":clip_rsm, "model_embedding":model_rsm}
_,_ = correlate_self(combined_rsm, n_permutation=0, multiple_comparison='fdr_bh', if_display=True, figsize=(14,13))


# Correlate combined RSM with neural RSM for each subject/group
corr_dfs = {}
for level in roi_neural_rsm.keys():
    print(level)
    print(f'\nCorrelating {level} with combined RSM:')
    corr_df = correlate_dicts_asymmetric(roi_neural_rsm[level], combined_rsm,
                                 n_permutation=0,
                                 multiple_comparison='fdr_bh') 
    plot_heatmap_asymmetric(corr_df)
    corr_dfs[level] = corr_df

# Average correlation values across subjects (excluding group)
subject_dfs = [corr_dfs[level] for level in corr_dfs.keys() if level != 'group']
average_df = subject_dfs[0].copy()

# Calculate mean correlation values
for i in range(len(average_df)):
    r_values = [df.iloc[i]['r'] for df in subject_dfs]
    average_df.iloc[i, average_df.columns.get_loc('r')] = np.mean(r_values)
    
    # Use majority vote for significance
    sig_signs = [df.iloc[i]['sig_sign'] for df in subject_dfs]
    most_common_sign = max(set(sig_signs), key=sig_signs.count)
    average_df.iloc[i, average_df.columns.get_loc('sig_sign')] = most_common_sign

print('\nAverage correlation across subjects:')
plot_heatmap_asymmetric(average_df)
#for each df (subject or average): (20 CLIP annotations + 4 model embeddings) * (11 RoIs * 2 sides) = 528

# %% 3. feedback: select candidates based on r
def select_candidates(df, r_threshold):
    """Select positive and negative candidates based on correlation values
    
    Args:
        df (pd.DataFrame): DataFrame containing correlation results
        r_threshold (float): Correlation threshold for splitting positive/negative examples
        
    Returns:
        dict: Dictionary mapping each reference to its positive/negative candidates
    """
    # Initialize output dictionary
    reference_dict = {}
    
    # Get unique references
    references = df['reference'].unique()
    
    # For each reference
    for ref in references:
        # Get correlations for CLIP annotations only
        ref_corrs = df[(df['reference'] == ref) & 
                      (df['module'] == 'CLIP_annotation')]
        
        # Split into positive and negative based on threshold
        positives = ref_corrs[ref_corrs['r'] >= r_threshold]['candidate'].tolist()
        negatives = ref_corrs[ref_corrs['r'] < r_threshold]['candidate'].tolist()
        
        # Store in dictionary
        reference_dict[ref] = {
            'positive': positives,
            'negative': negatives
        }
        
    return reference_dict

# Example usage
candidates_dict = select_candidates(average_df, r_threshold=0.1)
candidates_dict

# %%
def get_word_embedding(word, method='clip'):
    """Get semantic embedding for a word using specified method
    
    Args:
        word (str): Word to get embedding for
        method (str): Embedding method to use ('clip', 'openai', 'word2vec', 'bert')
    
    Returns:
        np.ndarray: Word embedding vector
    """
    if method == 'clip':
        # Use CLIP text encoder
        import clip
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load("ViT-B/32", device=device)
        text = clip.tokenize([word]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
        return text_features.cpu().numpy()[0]
    
    elif method == 'openai':
        # Use OpenAI text embeddings
        import openai
        response = openai.Embedding.create(
            input=word,
            model="text-embedding-ada-002"
        )
        return np.array(response['data'][0]['embedding'])
    
    elif method == 'word2vec':
        # Use Word2Vec
        import gensim.downloader
        model = gensim.downloader.load('word2vec-google-news-300')
        return model[word]
    
    elif method == 'bert':
        # Use BERT
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        inputs = tokenizer(word, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[0,0,:].numpy()
    
    else:
        raise ValueError(f"Unknown embedding method: {method}")

def get_weighted_candidates(df):
    """Extract correlation weights for CLIP annotation candidates for each reference
    
    Args:
        df (pd.DataFrame): DataFrame containing correlation results
        
    Returns:
        dict: Dictionary mapping each reference to its candidates and weights
    """
    # Initialize output dictionary
    weighted_dict = {}
    
    # Get unique references
    references = df['reference'].unique()
    
    # For each reference
    for ref in references:
        # Get correlations for CLIP annotations only
        ref_corrs = df[(df['reference'] == ref) & 
                      (df['module'] == 'CLIP_annotation')]
        
        # Create dictionary of candidates and their weights
        candidates = ref_corrs['candidate'].tolist()
        weights = ref_corrs['r'].tolist()
        
        weighted_dict[ref] = {
            'candidates': candidates,
            'weights': weights
        }
        
    return weighted_dict
