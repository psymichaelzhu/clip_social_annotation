
# file list





Question: Which dimension is represented by brain?
Model (semantic) representation vs. neural  representation

0. video preparation (matching)√ [match_videos.py]
1. video embedding vector√ [extract_video_embedding_CLIP.ipynb]
    videos
    embedding_df: CLIP_video 
         validate: within vs. between √ 
         description: 1) distribution √ 2) clustering √ [clustering_CLIP_embedding.ipynb]
2. neural data √
    check, organize
    neural activity -mask -summarize by roi
        which level
    more RoI? **whole-brain**
    specific RoI
 
3. construct rsm: √
    df
    rsm
        multiple: organize
    rsms (previous: neural, rating, CLIP overall, visual)

4. RSA: neural, CLIP, ResNet √
    CLIP is better than ResNet
    not satisfying, multiple dimensions--need clear separation

    statistical test

4. conditional annotation of video (dimensional) √
    concept, video_embedding_df
        dimension_list; multiple: softmax
    video_annotation_df
    validate:
        example | distribution, extreme
        human rating correlation

5. compare video annotation with neural √
    word, neural level
        word-video_annotation-video_annotation_df-video_annotation_rsm
        correlate with neural_rsm
    correlation coefficient (optimization goal)

6. go through design space
    define design space
        **discrete (from theory, LLM)**
        continuous embedding space, active learning
    for each word, run this 
        landscape
        summit

potential results
Region decodes .... (validate & discovery)



design space
GPT generate
    plot in semantic space (CLIP)


给定名字和description
建构list design space
相似性 优化
active loop


---
# design space转化为architecture space
# rsm
# 探索
#发现新的

载入



补充调整

其他embedding

相关/例子:
inference的潜力




# TODO
rating
    self: 对比DL
    neural: DNN强大
    整理-回归

函数 loop
LLM 
从一个GPT生成的小空间出发
基于语义去探索

innovation, memorability, 美学




embedding
1. 确定维度 (theory)
2. active loop: discover dimension


理论：
    social interaction相关
    不同脑区相关

    prompt组织

自动组合


词库: network, embedding

active learning： network prediction/embedding space wandering
sample-performance-funciton-new sample

语料库
少量出发 不断优化
潜力 展示




2 自动发掘 基于embedding
    **调研**
    **合并**

    **设计子维度**

    检查
    根据相关情况
    合并

    找到最佳维度

    解释: 相关

active loop

重要



how:
achtype: dimensional
where:
embedding hierarchical clustering....

简单的维度 (聚类)

文章

LLM



design space
how to choose dimension?
    active learning in embedding space:
        constrain: high-level (discrete), distance to social interaction (boundary)
    landscape

理论



network的形状
semantic的map






请列出所有与 "social interaction" 相关的概念，并根据其层次关系归类。
例如：
- 关系 (relationship): 亲子关系、朋友关系、权力关系
- 情感 (emotion): 亲密、信任、害怕、愤怒
- 交流 (communication): 语言交流、非语言交流、书面交流
- 场景 (scene): 工作场合、家庭环境、公共场所

扩展?



# 写作

先比CLIP和其他 找到优势模型
再来可解释性
open-window
可解释性 特异性



decomposite this into meaninful dimensions

future: human annotate, memorability 



now we have the pipeline: dimension (word) --> score (correlation)
    word -LLM prompting-> -CLIP encode-> CLIP embedding -cosine similarity-> dimensional annotation for video --> RSA (with neural) -rank correlation-> dimensional representational likelihood
    from a given dimension (a word), we can have the likelihood of neural activity encoding it.
with validations

dimension (word)
    single dimension
    multiple dimension (hierarchical structure)

then it becomes an optimization problem: find the dimension that yields highest likelihood, and that dimension is the one neural activity is encoding

the problem now is, what is the design space? **what are the potential dimensions we would like to examine?**
    if this space is large, we migh consider using the active learning loop to explore more efficiently. But let's start from the conceptual level, what would be the space?

few ways to define the design space:
1. theory-driven: synthesize previous literatures
    formalize: a concept into description
    **hierarchical organization: relationship = siblings + friends + couple ....**
        multiple dimensions: softmax: clasification task
        contrast

2. automatic semi-driven discovery: maybe find some unexpected dimension, like hand (isc 2004 science)
obtained from embedding space, like theory-driven, but without that kind of strong prior, using active learning to discover dimension

there are several **criterions** for this design space:
1. **videos are informative** --> the dimensions should have relatively wide distribution among videos 
2. taxtonomy: we are looking for dimensions, not instance? 
3. No. *shall we limit this to social interaction? relevant dimensions/irrelevant*

3. data-driven:
clustering the videos, find the potential dimensions (fishing....)
dimension reduction
clustering




# function

写一个函数
input: 
1. roi (roi_side的形式)
2. candidate_list (字典，表明了每个module里选哪些rsm)
3. candidate_rsm_dict (dict_dict 比如combined_rsm)
4. neural_rsm

从neural_rsm里提取subject在这个roi_side的rsm
再从candidate_rsm_dict中提取相应的

然后提取下三角

神经的做rank处理，特征的不处理

合并为df

sub video1 video2 roi feature1 feature2 ...

返回df


第二个函数
input：上面那个df
基于刚才这个df 建立回归模型
打印不同feature的回归系数
如果if_display
则展示不同feature的coefficient，从大到小(不包括intercept)
返回dict {roi: {intercept: intercept, feature1: beta1, feature2: beta2}}


    lmm 检查
    颜色  选择维度 颜色
    整合绘图

    consistency: rank; cross data
    mixed model
    bonferroni

    提mixed effect
    可视化




第二个函数
input: average_df 根据这里面的r 挑选那些r在0.2以上的roi-candidate组合
对于每一个roi, 都选择其所有的candidate
然后调用上面那个函数获得其回归系数
然后汇总所有roi的
返回dict
    汇总的显示形式
    统一的一组
    CLIP annotation + resnet50

