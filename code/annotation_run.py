# %% objective
# annotate videos using CLIP model
# whether from CLIP:https://github.com/openai/CLIP
# or OpenCLIP: https://github.com/mlfoundations/open_clip

# %% preparation

# packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image
from tqdm import tqdm

# set working directory to code directory
os.chdir('/Users/rezek_zhu/clip_social_annotation/code/new/clip_annotation')  
print(os.getcwd())

# model info
model_rank = [('openclip', 'ViT-H-14-378-quickgelu', 'dfn5b'),
              ('openclip', 'ViT-H-14-quickgelu', 'dfn5b'),
              ('openclip', 'ViT-bigG-14-quickgelu', 'metaclip_fullcc'),
              ('clip', 'ViT-L/14@336px', 'openai'),
              ('clip', 'ViT-B/32', 'openai'),
              ('openclip', 'ViT-bigG-14-CLIPA-336', 'datacomp1b')
]
model_rank_index = 0
model_source, model_name, pretrained_name = model_rank[model_rank_index]
print(model_source, model_name, pretrained_name)

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
if model_source == "openclip":
    #print("available models:", open_clip.list_pretrained())
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_name)
    model.eval() # switch to evaluation mode
    tokenizer = open_clip.get_tokenizer(model_name)
elif model_source == "clip":
    #print("available models:", clip.available_models())
    import clip
    model, preprocess = clip.load(model_name, device=device)
    tokenizer = clip.tokenize

# loading settings
overwrite = False
load_frame = True


# %% helper function
def extract_embedding(objects, type="image"):
    """
    Extract embedding using CLIP model based on input type

    input: 
        objects (image or text): a list of objects to extract embedding for. For image type, each element should be `numpy array` or `PIL image`. For text type, each element should be `string`.
        type (str): Type of objects, either "image" or "text"
    output: 
        embedding (tensor): Extracted features from CLIP model, which should be a tensor of shape (len(objects), embedding_dim).
    """
    if type == "image":
        image = [preprocess(obj if isinstance(obj, Image.Image) else Image.fromarray(obj)) for obj in objects] # list of tensors
        with torch.no_grad():
            features = model.encode_image(torch.stack(image).to(device))
    elif type == "text":
        text = tokenizer(objects).to(device)
        with torch.no_grad():
            features = model.encode_text(text)
    return features
    
def get_first_frame(video_path):
    """
    Extract the first frame from a video file and resize it to standard dimensions (300x250)

    input:
        video_path (str): Path to the video file
    output:
        frame (numpy array): First frame of the video in RGB format resized to standard dimensions, None if failed
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        # resize to standard dimensions
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (300, 250))
        return frame
    return None

# %% video
# video and frame embedding directory
embedding_dir = '../../../data/embedding/{}/{}/{}'.format(model_source.replace('/', '-'), model_name.replace('/', '-'), pretrained_name.replace('/', '-'))
print("embedding_dir:", embedding_dir)
os.makedirs(embedding_dir, exist_ok=True)

# video list
video_dir = '../../../data/video/original_clips'
video_name_list = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
#video_name_list = video_name_list[:2]#test

if os.path.exists(os.path.join(embedding_dir, 'video_embedding.npy')) and not overwrite:
    # Load existing embeddings
    print("Loading existing embeddings ...")    
    video_embedding = torch.from_numpy(np.load(os.path.join(embedding_dir, 'video_embedding.npy'))).to(device)
    if load_frame:
        frame_embeddings = np.load(os.path.join(embedding_dir, 'frame_embedding.npy'), allow_pickle=True)
        frame_embeddings = [emb.to(device) if isinstance(emb, torch.Tensor) else torch.from_numpy(emb).to(device) for emb in frame_embeddings]
    else:
        frame_embeddings = None
else:
    # store frame embeddings for each video
    frame_embeddings = []  # list of frame embeddings for each video
    checkpoint_interval = 30  # save every 30 videos
    
    for idx, video_name in enumerate(tqdm(video_name_list, desc="Processing videos")):
        # frame extraction
        video_path = os.path.join(video_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        # frame embedding
        frame_embedding = extract_embedding(frames, type="image")
        frame_embeddings.append(frame_embedding)

        # save checkpoint every 30 videos
        if (idx + 1) % checkpoint_interval == 0:
            print(f"\nSaving checkpoint at video {idx + 1}/{len(video_name_list)}...")
            
            # compute video embeddings for processed videos
            video_embedding_checkpoint = torch.zeros((len(frame_embeddings), model.visual.output_dim)).to(device)
            for i, emb in enumerate(frame_embeddings):
                video_embedding_checkpoint[i] = torch.mean(emb, dim=0)
                
            # save checkpoints
            np.save(os.path.join(embedding_dir, f'video_embedding_checkpoint_{idx+1}.npy'), 
                   video_embedding_checkpoint.cpu().numpy())
            np.save(os.path.join(embedding_dir, f'frame_embedding_checkpoint_{idx+1}.npy'), 
                   np.array(frame_embeddings, dtype=object))

    # compute final video embeddings
    video_embedding = torch.zeros((len(video_name_list), model.visual.output_dim)).to(device)
    for i, frame_embedding in enumerate(frame_embeddings):
        video_embedding[i] = torch.mean(frame_embedding, dim=0)

    # save final video and frame embeddings
    print("\nSaving final embeddings...")
    np.save(os.path.join(embedding_dir, 'video_embedding.npy'), video_embedding.cpu().numpy())
    np.save(os.path.join(embedding_dir, 'frame_embedding.npy'), np.array(frame_embeddings, dtype=object))

# video embedding matrix: video by embedding_dim
print(video_embedding.shape)
if frame_embeddings is not None:
    print(len(frame_embeddings))
    print(frame_embeddings[0].shape)

# %% validation of video embedding aggregation: 
if frame_embeddings is not None:
    # 1) frame-video similarity
    frame_similarity = []
    cos = torch.nn.CosineSimilarity(dim=1)
    for i, frame_embedding in enumerate(frame_embeddings):
        similarities = cos(frame_embedding, video_embedding[i].unsqueeze(0).repeat(frame_embedding.shape[0], 1))
        frame_similarity.append(similarities)

    # distributions of min/mean/max similarities
    min_sims = [sim.min().item() for sim in frame_similarity]
    mean_sims = [sim.mean().item() for sim in frame_similarity]
    max_sims = [sim.max().item() for sim in frame_similarity]

    # lower bound selectionrationale
    if True:
        lower_bound = 0.9
        #bins = np.arange(min(min_sims), 1.01, 0.01)
        below_threshold = sum(1 for sim in min_sims if sim < lower_bound)
        print(f"A few outliers ({below_threshold/len(min_sims)*100:.2f}%) fall below {lower_bound}, omitted here for clarity.")

    plt.figure(figsize=(9,6))
    bins=np.arange(lower_bound, 1.0, 0.01)
    plt.hist(min_sims, bins=bins, alpha=0.3, label='Minimum', color='#1b9e77')
    plt.hist(mean_sims, bins=bins, alpha=0.3, label='Mean', color='#d95f02')
    plt.hist(max_sims, bins=bins, alpha=0.3, label='Maximum', color='#7570b3')

    plt.title('Frame-Video Embedding Similarity Distribution', fontsize=22)
    plt.xlabel('Frame-Video Cosine Similarity', fontsize=18)
    plt.ylabel('Count', fontsize=18)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=18)

    plt.tight_layout()
    plt.show()

# %% prompt
# prompt list
prompt_list = ['indoor rather than outdoor scene', # indoor
                'people acting in a near space rather than far space', # expanse
                'someone interacting with an object', # transitivity
                'people physically close to one another rather than far apart', # agent_distance
                'people facing each other', # facingness
                'people acting jointly rather than independently', # joint_action
                'people communicating', # communication
                'a pleasant rather than unpleasant action', # valence
                'an emotionally intense/arousing rather than calm action'] # arousal

prompt_alias = ['indoor', 'expanse', 'transitivity', 'agent_distance', 'facingness', 'joint_action', 'communication', 'valence', 'arousal']

# dimension categories and colors from paper
dimension_categories = {
    "Scene & Object": ["indoor", "expanse", "transitivity"],
    "Social Primitive": ["agent_distance", "facingness"],
    "Social Interaction": ["joint_action", "communication"],
    "Affective": ["valence", "arousal"]
}
category_colors = {
    "Scene & Object": "#F9ED8E",
    "Social Primitive": "#C0ABF8",
    "Social Interaction": "#B2E7EE",
    "Affective": "#E2ACAE"
}

# %% annotation attempt 1: use original prompt embedding | not helpful
prompt_embedding = extract_embedding(prompt_list, type="text")

# compute cosine similarity between each video and prompt embedding
video_np = video_embedding.cpu().numpy()  # (video_num, embedding_dim)
prompt_np = prompt_embedding.cpu().numpy()  # (prompt_num, embedding_dim)

from sklearn.metrics.pairwise import cosine_similarity
cos_sim = cosine_similarity(video_np, prompt_np)

# cosine similarity matrix: video by prompt
print(cos_sim.shape)


# %% annotation attempt 2: add "A picture of" to the prompt | not helpful
video_np = video_embedding.cpu().numpy() 
prompt_list_with_prefix = ['A picture of'+p for p in prompt_list]
prompt_embedding_with_prefix = extract_embedding(prompt_list_with_prefix, type="text")

cos_sim = cosine_similarity(video_np, prompt_embedding_with_prefix.cpu().numpy())
print(cos_sim.shape)

# %% annotation attempt 3: use bipolar annotation ｜ working well!
bipolar_prompts = {
    'indoor': ['an indoor scene', 'an outdoor scene'],
    'expanse': ['people acting in a near space', 'people acting in a far space'], 
    'transitivity': ['someone interacting with an object', 'someone not interacting with any object'],
    'agent_distance': ['people physically close to one another', 'people physically far apart'],
    'facingness': ['people facing each other', 'people facing away from each other'],
    'joint_action': ['people acting jointly', 'people acting independently'],
    'communication': ['people communicating with each other', 'people not communicating'],
    'valence': ['a pleasant action', 'an unpleasant action'],
    'arousal': ['an emotionally intense/arousing action', 'a calm action']
}

# Get embeddings for both positive and negative prompts
pos_prompts = ['A picture of '+v[0] for v in bipolar_prompts.values()] #prompting trick for clip model, see https://github.com/openai/CLIP?tab=readme-ov-file#zero-shot-prediction
neg_prompts = ['A picture of '+v[1] for v in bipolar_prompts.values()]

pos_embedding = extract_embedding(pos_prompts, type="text").cpu().numpy()
neg_embedding = extract_embedding(neg_prompts, type="text").cpu().numpy()

video_np = video_embedding.cpu().numpy()

pos_sim = cosine_similarity(video_np, pos_embedding)
neg_sim = cosine_similarity(video_np, neg_embedding)

annotation_algorithm_index = 2
annotation_algorithms = ['softmax', 'constrasting', 'embedding_algebra']
# 1 and 2 are foundamentally the same (different scaling), 3 is slightly different (best)

if annotation_algorithm_index == 0:
    # rating algorithm 1: softmax
    print("rating algorithm 1: softmax")
    class_sims = np.stack([pos_sim, neg_sim], axis=2)
    from scipy.special import softmax
    probs = softmax(class_sims, axis=2)
    cos_sim = probs[:,:,0]
elif annotation_algorithm_index == 1:
    # rating algorithm 2: constrasting positive and negative similarities
    print("rating algorithm 2: constrasting positive and negative similarities")
    cos_sim = (pos_sim - neg_sim) / 2
elif annotation_algorithm_index == 2:
    # rating algorithm 3: embedding algebra
    print("rating algorithm 3: embedding algebra")
    diff_embedding = pos_embedding - neg_embedding 
    cos_sim = cosine_similarity(video_np, diff_embedding)
else:
    raise ValueError(f"Invalid annotation algorithm index: {annotation_algorithm_index}")

# Reverse code expanse and agent_distance
expanse_idx = list(bipolar_prompts.keys()).index('expanse')
agent_distance_idx = list(bipolar_prompts.keys()).index('agent_distance')
cos_sim[:,expanse_idx] = 1 - cos_sim[:,expanse_idx]
cos_sim[:,agent_distance_idx] = 1 - cos_sim[:,agent_distance_idx]

print("Final annotation matrix shape (videos x prompts):", cos_sim.shape)

# %% annotation follow-up:
# 1) validation: distribution of model annotation 
# An intuitive presentation to let the audience see if the model annotations make sense.
def plot_model_annotation_distribution(cos_sim_df, prompt_list, prompt_alias, video_dir, video_name_list, n_cols=3):
    """
    Plot distribution of model annotations with example frames at different percentiles

    input:
        cos_sim_df (numpy array or pandas DataFrame): Cosine similarities between videos and prompts
        prompt_list (list): List of prompt strings
        prompt_alias (list): List of short alias strings for prompts
        video_dir (str): Directory containing the video files
        video_name_list (list): List of video filenames
        n_cols (int): Number of columns in the subplot grid
    output:
        None: Displays the plot
    """
    # Convert numpy array to DataFrame if needed
    if isinstance(cos_sim_df, np.ndarray):
        cos_sim_df = pd.DataFrame(cos_sim_df, columns=prompt_alias)

    # Create subplots for each prompt dimension
    n_prompts = len(prompt_alias)
    n_rows = (n_prompts + n_cols - 1) // n_cols  # Ceiling division
    fig = plt.figure(figsize=(6*n_cols, 4.5*n_rows))

    plt.suptitle('Distribution of Model Annotation (Video-Prompt Similarity)', fontsize=34)

    for i, prompt in enumerate(prompt_alias):
        ax = plt.subplot(n_rows, n_cols, i+1)
        
        # Get category for current prompt
        category = next(cat for cat, dims in dimension_categories.items() if prompt in dims)
        hist_color = category_colors[category]
        
        # Plot histogram of similarities for this prompt
        similarities = cos_sim_df[prompt].values
        plt.hist(similarities, bins=30, color=hist_color, alpha=0.7)
        
        # Get videos at different percentiles
        percentiles = [0, 25, 50, 75, 100]
        sorted_indices = np.argsort(similarities)
        n = len(sorted_indices)
        percentile_indices = [sorted_indices[int((p/100.0) * (n-1))] for p in percentiles]
        
        # Add images and vertical lines for selected percentiles
        for j, (idx, p) in enumerate(zip(percentile_indices, percentiles)):
            sim_value = similarities[idx]
            
            plt.axvline(x=sim_value, color='#1b9e77', linestyle='--', alpha=0.5)
            
            # Get first frame for this video
            video_path = os.path.join(video_dir, video_name_list[idx])
            frame = get_first_frame(video_path)
            if frame is not None:
                img_width = 0.3  # Width for landscape format
                img_height = 0.25  # Height for landscape format
                trans = ax.transData.transform([(sim_value, 0)])[0]
                trans = ax.transAxes.inverted().transform(trans)
                img_x = trans[0] - img_width/2
                # Adjust x position for first and last percentile images
                if j == 0:
                    img_x += img_width/4
                elif j == len(percentiles)-1:
                    img_x -= img_width/4
                # Increased vertical spacing between images
                img_y = 0.45 if j % 2 == 0 else 0.15
                
                inset = ax.inset_axes([img_x, img_y, img_width, img_height])
                
                inset.imshow(frame)
                inset.axis('off')
        
        plt.title(f'{prompt}', fontsize=28)
        plt.xlabel('Cosine Similarity', fontsize=22)
        plt.ylabel('Count', fontsize=22)
        plt.tick_params(labelsize=20)

    plt.tight_layout()
    plt.show()

plot_model_annotation_distribution(cos_sim, prompt_list, prompt_alias, video_dir, video_name_list, n_cols=3)

# 2) save scaled scores as csv
# min-max scale to 0-1 range to match human rating range
cos_sim_scaled = (cos_sim - cos_sim.min(axis=0)) / (cos_sim.max(axis=0) - cos_sim.min(axis=0))

annotation_dir = '../../../data/annotation/{}/{}/{}'.format(model_source.replace('/', '-'), model_name.replace('/', '-'), pretrained_name.replace('/', '-'))
print("annotation_dir:", annotation_dir)
os.makedirs(annotation_dir, exist_ok=True)
cos_sim_scaled = pd.DataFrame(cos_sim_scaled, columns=prompt_alias, index=video_name_list)  
cos_sim_scaled.index.name = 'video_name'
cos_sim_scaled.to_csv(os.path.join(annotation_dir, 'model_annotation.csv'), index=True)


#%%
# 3) Validation: comparison with human annotation results
# in R: validation_annotation_correlation.R



# Test get_first_frame function on one video to check frame dimensions
test_video_path = os.path.join(video_dir, video_name_list[0])
frame = get_first_frame(test_video_path)
if frame is not None:
    print(f"Frame shape (height, width, channels): {frame.shape}")
else:
    print("Failed to extract first frame")

# %%