The initial motivation of this project is to establish a mapping from theoretical dimensions to brain regions in social interaction through a novel and efficient annotation pipeline.

Our contributions will be divided into two parts:
1. Novel annotation process: Efficient annotation using multimodal models
2. "Meta-analysis": Integration of datasets and dimensions of social interaction

We start testing the feasibility from the open dataset of (MacMahon et al., 2023).

However, after exploring different multi-modality models, annotation strategies, and prompts,

the best results still fall short of capturing human ratings, especially in relatively high-level dimensions such as "joint action" or "arousal".

Consistent with the study of Garcia et al., 2024, they found that LLMs and visual models (including CLIP2) cannot reliably represent human ratings on all dimensions, and none of them can significantly explain neural patterns.

We assume that this is the result of inherent bottlenecks in multimodal models: these models are mainly trained on online labeled data, and people often use referential language (e.g. "this is my dog") rather than high-level language (e.g. "these two people are acting together") to describe.
