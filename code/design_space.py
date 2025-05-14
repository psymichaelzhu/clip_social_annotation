#%% packages
import nltk
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertModel
import torch
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

#%% function
def extract_hierarchical_concepts(seed_words, max_depth=2):
    """
    Extract hierarchical concepts from WordNet starting from seed words
    
    Args:
        seed_words (list): List of starting words to extract hierarchy from
        max_depth (int): Maximum depth of hierarchy to explore
        
    Returns:
        dict: A hierarchical dictionary containing word relationships
    """
    def get_related_concepts(synset, current_depth):
        if current_depth >= max_depth:
            return {}
        
        result = {}
        
        # Get hyponyms (more specific concepts)
        for hyponym in synset.hyponyms():
            result[hyponym.lemmas()[0].name()] = get_related_concepts(hyponym, current_depth + 1)
            
        # Get hypernyms (more general concepts)
        for hypernym in synset.hypernyms():
            result[hypernym.lemmas()[0].name()] = get_related_concepts(hypernym, current_depth + 1)
            
        # Get also_sees (related concepts)
        for also_see in synset.also_sees():
            result[also_see.lemmas()[0].name()] = get_related_concepts(also_see, current_depth + 1)
            
        return result

    hierarchy = {}
    for seed_word in seed_words:
        synsets = wn.synsets(seed_word)
        for synset in synsets:
            name = synset.lemmas()[0].name()
            if name not in hierarchy:  # Avoid duplicates
                hierarchy[name] = get_related_concepts(synset, 0)
        
    return hierarchy

def filter_concepts_by_similarity(hierarchy, target_words, threshold=0.3):
    """
    Filter concepts based on BERT embedding similarity with target words
    
    Args:
        hierarchy (dict): Hierarchical dictionary of concepts
        target_words (list): Words to compare similarity against
        threshold (float): Minimum similarity threshold
        
    Returns:
        dict: Filtered hierarchical dictionary
    """
    # Initialize BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    
    def get_bert_embedding(word):
        inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()
    
    # Get target embeddings
    target_embeddings = [get_bert_embedding(word) for word in target_words]
    
    def filter_dict(d):
        filtered = {}
        for key, value in d.items():
            word_embedding = get_bert_embedding(key)
            
            # Calculate maximum similarity with any target word
            max_similarity = max(
                torch.cosine_similarity(word_embedding, target_emb, dim=0)
                for target_emb in target_embeddings
            )
            
            if max_similarity > threshold:
                if isinstance(value, dict):
                    filtered_children = filter_dict(value)
                    if filtered_children:
                        filtered[key] = filtered_children
                else:
                    filtered[key] = value
                    
        return filtered
    
    return filter_dict(hierarchy)

def visualize_hierarchy(hierarchy, output_file='hierarchy.png'):
    """
    Visualize hierarchical concepts using networkx
    
    Args:
        hierarchy (dict): Hierarchical dictionary to visualize
        output_file (str): Path to save the visualization
        
    Returns:
        None: Saves visualization to file
    """
    G = nx.DiGraph()  # Changed to DiGraph for directed edges
    
    def add_nodes_edges(d, parent=None):
        for key, value in d.items():
            # Clean up the key for display
            display_key = key.replace('_', ' ')
            G.add_node(display_key)
            if parent:
                G.add_edge(parent, display_key)
            if isinstance(value, dict):
                add_nodes_edges(value, display_key)
    
    add_nodes_edges(hierarchy)
    
    if len(G.nodes) == 0:
        print("Warning: No nodes to visualize!")
        return
    
    plt.figure(figsize=(15, 10))
    plt.rcParams.update({'font.size': 18})
    
    try:
        pos = nx.spring_layout(G, k=2, iterations=50)  # Increased k for more spread
        nx.draw(G, pos, 
                with_labels=True, 
                node_color='lightblue',
                node_size=2000, 
                font_size=8,  # Reduced font size
                font_weight='bold',
                arrows=True,  # Show direction
                edge_color='gray',
                width=2)
        
        plt.title('Social Interaction Concept Hierarchy', fontsize=22)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Error in visualization: {e}")
    finally:
        plt.close()

#%% Example usage
if __name__ == "__main__":
    # Make sure we have the WordNet data
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    # Define seed words and target words
    seed_words = ['social', 'interaction', 'communicate', 'relationship']
    target_words = ['social']
    
    print("Extracting concepts...")
    hierarchy = extract_hierarchical_concepts(seed_words)
    
    print("Filtering concepts...")
    filtered_hierarchy = filter_concepts_by_similarity(hierarchy, target_words)
    
    print("Visualizing results...")
    visualize_hierarchy(filtered_hierarchy, 'social_interaction_hierarchy.png')
    
    # Print the hierarchy structure
    print("\nHierarchy structure:")
    def print_hierarchy(d, level=0):
        for k, v in d.items():
            print("  " * level + f"- {k}")
            if isinstance(v, dict):
                print_hierarchy(v, level + 1)
    
    print_hierarchy(filtered_hierarchy)

#%%
filtered_hierarchy

# %%
extract_hierarchical_concepts(['social'])

# %%
