`template_formatting.py`: a demo template for the general format of the code

# Video
`match_videos.py`: match and copy videos used in the study from  the original Moments in Time dataset

# Neural
`utils/load_roi_activation.py`: function to load the activation data with RoI mask (load_neural_data)
`extract_neural_activation.py`: organize neural activation data |nested dictionary, subject - ROI - side: video x voxel (beta)

# DNN
## model embedding
video x embedding dimension, embedding value
### CLIP
`extract_clip_embedding.py`: extract CLIP embeddings for videos

### ResNet
`extract_resnet_embedding.py`: extract ResNet embeddings for videos

## dimension annotation
`CLIP_dimension_annotate.py`: annotate the dimensions of the videos (e.g. 'relationship'->'parent-child', 'couple', 'coworkers', 'friends','neighbors') | dictionary, dimension: video x label (normalized similarity score)

prerequiste: `extract_clip_embedding.py`







can't iterate all labels?
use several labels, extract node activation
compare the activation with neural (embedding extraction, language)

dimension, different labels define a dimension
similarity on this dimension

between-video similarity