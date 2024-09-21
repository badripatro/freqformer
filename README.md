# FreqFormer: Optimizing Vision Transformers with Frequency and Attention


## Abstract

'''
In this study, we introduce FreqFormer, a novel architecture for vision transformers that optimizes performance by integrating frequency and attention mechanisms. Vision transformers have demonstrated significant success in image recognition tasks, with models like ViT and DeIT utilizing multi-headed self-attention, and others like Fnet, GFNet, and AFNO leveraging spectral layers. We hypothesize that spectral layers effectively capture high-frequency information such as lines and edges, while attention layers excel at capturing token interactions. Our investigation confirms that combining spectral and multi-headed attention layers results in a superior transformer architecture. FreqFormer, with its initial spectral and deeper multi-headed attention layers, achieves improved performance over existing transformer models. Notably, FreqFormer-H-S attains a top-1 accuracy of 84.25\% on ImageNet-1K, setting a new state-of-the-art for small versions, while FreqFormer-H-L reaches 85.7\%, leading the comparable base versions. We further validate FreqFormer's performance through transfer learning on standard datasets like CIFAR-10, CIFAR-100, Oxford-IIIT-flower, and Stanford Car, as well as in downstream tasks such as object detection and instance segmentation on the MS-COCO dataset. Our results demonstrate that FreqFormer consistently delivers competitive performance, highlighting its potential for further optimization and improvement.

'''



## Training

### Train FreqFormer for Vanilla Architecture 
```
bash vanilla_architecture/main.sh
```


### Train FreqFormer for Hierarchical Architecture 
```
bash hierarchical_architecture/main.sh
```

