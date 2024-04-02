# UniMD: Towards Unifying Moment retrieval and temporal action Detection



# video feature

We provide the video features used in our experiments, and explain their extraction in three paired datasets that includes Temporal Action Detection (TAD) and Moment Retrieval (MR). The datasets includes "Ego4D-MD and Ego4D-NLQ", "Charades and Charades-STA", and ActivityNet and ActivityNet-caption.

* **Ego4D-MQ & Ego4D-NLQ:** we use the features provided by InternVideo-ego4D. Their features are 1024-D and based on clips of 16 frames, with a frame rate of 30 and a stride of 16 frames. <u>It is important to note</u> that these features are extracted by VideoMAE-L, pretrained on the verb and noun subset without second finetuning stage, such as "K700→ Verb" in the paper. The features can be available in the section "Video Features for MQ and NLQ" in [InternVideo-ego4D](https://github.com/OpenGVLab/ego4d-eccv2022-solutions). 
* **Charades & Charades-STA:** we use I3D and CLIP features as the input video features. The I3D features are extracted from snippets with a window of 16 frames and a stride of 4. Each snippet yields two types of 1024-Dfeatures: one based on RGB frames decoded at 24 fps, and the other based on optical flow. The I3D features are formed by concatenating the RGB features and optical flow features, resulting in a 2048-D feature. In addition, we extract CLIP features every 4 frames using CLIP-B. 
  * I3D features download: we extract I3D features following the [method](https://github.com/Finspire13/pytorch-i3d-feature-extraction), and they can be downloaded in [google drive]().
  * CLIP features download: the features can be downloaded in [google drive]().

* **ActivityNet & ActivityNet-caption:** we follow the [InternVideo](https://arxiv.org/pdf/2212.03191.pdf) for feature extraction, the backbone of which is named InternVideo and based on ViT-H. Each step of the video features is extracted from a continuous 16 frames with a stride of 16 in 30 fps video. The features can be downloaded in [here](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo1/Downstream/Temporal-Action-Localization). The TSP features can be downloaded in [here](https://github.com/happyharrycn/actionformer_release?tab=readme-ov-file#to-reproduce-our-results-on-activitynet-13).

# query embeddings & gt

* **Ego4D:** In Ego4D-MQ (TAD task), we use the category name itself as input to extract text embeddings, without additional prefixes/suffixes/learnable prompts. In NLQ (MR task) we use natural language descriptions from annotations. The text embeddings of queries as well as groundtruth can be downloaded in [google drive]().
* **Charades & Charades-STA**: the formation method of text embeddings is the same as Ego4d, which can be downloaded in [google drive]().
* **ActivityNet & ActivityNet-caption:** in TAD task, we firstly draw text embeddings of whole 200 categories, and then average the normalized text embeddings of whole categories for action proposals. When inferring, we use external video action recognition to obtain specific action categories. In MR, the procedure of drawing text embedding is the same as the Ego4D and Charades. Text embeddings and GT can be downloaded in [google drive]().

## Acknowledgement





# Citation

