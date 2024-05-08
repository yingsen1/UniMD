# UniMD: Towards Unifying Moment Retrieval and Temporal Action Detection

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unimd-towards-unifying-moment-retrieval-and/natural-language-queries-on-ego4d)](https://paperswithcode.com/sota/natural-language-queries-on-ego4d?p=unimd-towards-unifying-moment-retrieval-and)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unimd-towards-unifying-moment-retrieval-and/moment-queries-on-ego4d)](https://paperswithcode.com/sota/moment-queries-on-ego4d?p=unimd-towards-unifying-moment-retrieval-and)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unimd-towards-unifying-moment-retrieval-and/moment-retrieval-on-charades-sta)](https://paperswithcode.com/sota/moment-retrieval-on-charades-sta?p=unimd-towards-unifying-moment-retrieval-and)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unimd-towards-unifying-moment-retrieval-and/action-detection-on-charades)](https://paperswithcode.com/sota/action-detection-on-charades?p=unimd-towards-unifying-moment-retrieval-and)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unimd-towards-unifying-moment-retrieval-and/temporal-action-localization-on-activitynet)](https://paperswithcode.com/sota/temporal-action-localization-on-activitynet?p=unimd-towards-unifying-moment-retrieval-and)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unimd-towards-unifying-moment-retrieval-and/natural-language-moment-retrieval-on)](https://paperswithcode.com/sota/natural-language-moment-retrieval-on?p=unimd-towards-unifying-moment-retrieval-and)  

[\[arxiv\]](https://arxiv.org/abs/2404.04933)  

<div align="center">
  <img src="./images/intro_unimd.png" width="600px"/>
</div>

<div align="center">
  <img src="./images/network.png" width="600px"/>
</div>

## Introduction

In this paper, we aim to investigate the potential synergy between TAD and MR. Firstly, we propose a unified architecture, termed Unified Moment Detection (UniMD), for both TAD and MR, as shown in Fig.1. It transforms the inputs of the two tasks, namely actions for TAD or events for MR, into a common embedding space, and utilizes two novel querydependent decoders to generate a uniform output of classification score and temporal segments, as shown in Fig.4. Secondly, we explore the efficacy of two task fusion learning approaches, pre-training and co-training, in order to enhance the mutual benefits between TAD and MR. Extensive experiments demonstrate that the proposed task fusion learning scheme enables the two tasks to help each other and outperform the separately trained counterparts.

This repository will contain the code for UniMD and the video features used in the paper. Our code is built upon the codebase from [actionformer](https://github.com/happyharrycn/actionformer_release).



## Changelog

* 29/04/2024: release the inference & train code.
* 02/04/2024: we create the code repo and release the video features, text embeddings and groundtruth used in the paper.

## Video Features

We provide the video features of three paired datasets used in our experiments, including "Ego4D-MQ & Ego4D-NLQ", "Charades & Charades-STA", and "ActivityNet & ActivityNet-caption".

* **Ego4D-MQ & Ego4D-NLQ:** we use the features provided by InternVideo-ego4D. Their features are 1024-D and based on clips of 16 frames, with a frame rate of 30 and a stride of 16 frames. <u>It is important to note</u> that these features are extracted by VideoMAE-L, pretrained on the verb and noun subset without second finetuning stage, such as "K700→ Verb" in the InternVideo-ego4D paper. The features can be available in the section "Video Features for MQ and NLQ" in [InternVideo-ego4D](https://github.com/OpenGVLab/ego4d-eccv2022-solutions). 
* **Charades & Charades-STA:** we use I3D and CLIP features as the input video features. The I3D features are extracted from snippets with a window of 16 frames and a stride of 4. Each snippet yields two types of 1024-Dfeatures: one based on RGB frames decoded at 24 fps, and the other based on optical flow. The I3D features are formed by concatenating the RGB features and optical flow features, resulting in a 2048-D feature. In addition, we extract CLIP features every 4 frames using CLIP-B. 
  * I3D: we extract I3D features following the [method](https://github.com/Finspire13/pytorch-i3d-feature-extraction), and they can be downloaded in [google drive](https://drive.google.com/file/d/1jivG3olcvhayAxE7waec7HoRwxFtJTlT/view?usp=sharing).
  * CLIP: features can be downloaded in [google drive](https://drive.google.com/file/d/1WtqmkPMS5kiXe9RFL6xYOEVdE8jle-2P/view?usp=sharing).

* **ActivityNet & ActivityNet-caption:** we follow the [InternVideo](https://arxiv.org/pdf/2212.03191.pdf) for feature extraction, the backbone of which is named InternVideo and based on ViT-H. Each step of the video features is extracted from a continuous 16 frames with a stride of 16 in 30 fps video. The features can be downloaded in [here](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo1/Downstream/Temporal-Action-Localization). The TSP features can be downloaded in [here](https://github.com/happyharrycn/actionformer_release?tab=readme-ov-file#to-reproduce-our-results-on-activitynet-13).

## query embeddings & groundtruth

* **Ego4D:** In Ego4D-MQ (TAD), we use the category name itself as input to extract text embeddings, without additional prefixes/suffixes/learnable prompts. In NLQ (MR) we use natural language descriptions from annotations. The text embeddings of queries as well as groundtruth can be downloaded in [google drive](https://drive.google.com/file/d/1Kefm4NtAdf3KIsMskvK5QpwGB1t2uwbA/view?usp=sharing).
* **Charades & Charades-STA**: the formation method of text embeddings is the same as Ego4d, which can be downloaded in [google drive](https://drive.google.com/file/d/1_Piovqjal4FD8NiHc6b9YadL0qeHLrT4/view?usp=sharing).
* **ActivityNet & ActivityNet-caption:** in TAD, we firstly draw text embeddings of whole 200 categories, and then average the normalized text embeddings of whole categories for action proposals. When inferring, we use external video action recognition to obtain specific action categories. In MR, the procedure of drawing text embedding is the same as the Ego4D and Charades. Text embeddings and GT can be downloaded in [google drive](https://drive.google.com/file/d/1RjCQkC1OkZpb-f8c1xCQAHjBLRzOCRnM/view?usp=drive_link).

## Installation

* Follow [INSTALL.md](./INSTALL.md) for installing necessary dependencies and compiling the code.

## Inference

### Ego4D

| Dataset   | Method     | Feats.         | TAD-mAP | TAD-r1@50 | MR-r1@30 | MR-r1@50 | checkpoint                                                   |
| --------- | ---------- | -------------- | ------- | --------- | -------- | -------- | ------------------------------------------------------------ |
| Ego4D-MQ  | individual | InternVid-verb | 22.61   | 42.82     | -        | -        | [ego4d_mq_individual.pth.tar](https://drive.google.com/file/d/16AMkzP9L606uZj2d-M1IFEi6fmBRR3wB/view?usp=drive_link) |
| Ego4D-NLQ | individual | InternVid-verb | -       | -         | 13.99    | 9.34     | [ego4d_nlq_individual.pth.tar](https://drive.google.com/file/d/1JfJmnejKNEbsnpDxsnyUy_QN_SMlCz-a/view?usp=drive_link) |

- we wil release cotrained checkpoints soon.
- download the checkpoint and place it in *./checkpoint/individual/ego4d/*. For TAD task, make data_type=tad in shell script. For MR task, make data_type=mr. For both task simultaneously, make data_type=all.

```
# inference
cd ./tools/
sh run_predict_ego4d.sh     # for tad: make data_type=tad; for mr: make data_type=mr
```

### Charades & Charades-STA

| Dataset      | Method     | feats    | TAD-mAP | MR-r1@50 | MR-r1@70 | checkpoint                                                   |
| ------------ | ---------- | -------- | ------- | -------- | -------- | ------------------------------------------------------------ |
| Charades     | individual | i3d      | 22.31   | -        | -        | [charades_individual.pth.tar](https://drive.google.com/file/d/1PftFrX7XSzhAlERParIHNMdLgN9D-WiI/view?usp=drive_link) |
| Charades     | individual | i3d+clip | 26.18   | -        |          | [charades_i3dClip_individual.pth.tar](https://drive.google.com/file/d/1n9WNU9PQU9f0SAjKrHVzEfjVfgOxYUhK/view?usp=drive_link) |
| Charades-STA | individual | i3d      | -       | 60.19    | 41.02    | [charadesSTA_individual.pth.tar](https://drive.google.com/file/d/1qLrUQU1NDLvJzkXzyXM3WklL_8IXtqjw/view?usp=drive_link) |
| Charades-STA | individual | i3d+clip | -       | 58.79    | 40.08    | [charadesSTA_i3dClip_individual.pth.tar](https://drive.google.com/file/d/15j15o_gVk42SbYPV3bqEWjFdxKCHLQRk/view?usp=drive_link) |

* download the checkpoint and place it in *./checkpoint/individual/charades/*.

```
# inference
cd ./tools/
sh run_predict_charades.sh     # for tad: make data_type=tad; for mr: make data_type=mr
```

### ActivityNet & ActivityNet-Caption

| Dataset      | Method     | feats     | TAD-mAP | TAD-mAP@50 | MR-r5@50 | MR-r5@70 | checkpoint                                                   |
| ------------ | ---------- | --------- | ------- | ---------- | -------- | -------- | ------------------------------------------------------------ |
| ANet         | individual | internVid | 38.60   | 58.31      | -        | -        | [anet_tad_individual.pth.tar](https://drive.google.com/file/d/1qmSud7da1CaidjmM0MXZmQWeVrJUzeqB/view?usp=drive_link) |
| ANet-caption | individual | internVid | -       | -          | 77.28    | 52.22    | [anet_caption_individual.pth.tar](https://drive.google.com/file/d/1e341eeaUbRn1mamTrMlnsPbs30aJnrJI/view?usp=drive_link) |

* download the checkpoint and place it in *./checkpoint/individual/anet/*.

```
# inference
cd ./tools/
sh run_predict_anet.sh     # for tad: make data_type=tad; for mr: make data_type=mr
```



## Training

- individual train

```
cd ./tools/
# for example, train ego4d in tad task individually, run:
sh individual_train_ego4d_tad.sh
```

- co-train (random sampling)

```
cd ./tools/
# forexample, co-train ego4d, run:
sh cotrain_random_ego4d.sh
```



## Citation

```
@misc{zeng2024unimd,
      title={UniMD: Towards Unifying Moment Retrieval and Temporal Action Detection}, 
      author={Yingsen Zeng and Yujie Zhong and Chengjian Feng and Lin Ma},
      year={2024},
      eprint={2404.04933},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Acknowledgement

This repository is built based on [ActionFormer](), [InternVideo-ego4d](https://github.com/OpenGVLab/ego4d-eccv2022-solutions), [InternVideo](https://github.com/OpenGVLab/InternVideo?tab=readme-ov-file), [i3d-feature-extraction](https://github.com/Finspire13/pytorch-i3d-feature-extraction) repository.
