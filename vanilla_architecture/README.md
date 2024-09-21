# FreqFormer Model for Image Classification

## Abstract 

'''
In this study, we introduce FreqFormer, a novel architecture for vision transformers that optimizes performance by integrating frequency and attention mechanisms. Vision transformers have demonstrated significant success in image recognition tasks, with models like ViT and DeIT utilizing multi-headed self-attention, and others like Fnet, GFNet, and AFNO leveraging spectral layers. We hypothesize that spectral layers effectively capture high-frequency information such as lines and edges, while attention layers excel at capturing token interactions. Our investigation confirms that combining spectral and multi-headed attention layers results in a superior transformer architecture. FreqFormer, with its initial spectral and deeper multi-headed attention layers, achieves improved performance over existing transformer models. Notably, FreqFormer-H-S attains a top-1 accuracy of 84.25\% on ImageNet-1K, setting a new state-of-the-art for small versions, while FreqFormer-H-L reaches 85.7\%, leading the comparable base versions. We further validate FreqFormer's performance through transfer learning on standard datasets like CIFAR-10, CIFAR-100, Oxford-IIIT-flower, and Stanford Car, as well as in downstream tasks such as object detection and instance segmentation on the MS-COCO dataset. Our results demonstrate that FreqFormer consistently delivers competitive performance, highlighting its potential for further optimization and improvement.

'''



## Model Zoo

We provide variants of our FreqFormer models trained on ImageNet-1K:
| name |  Params | FLOPs | acc@1 | acc@5 | 
| --- | --- | --- | --- | --- | 
| FreqFormer-T | 9M | 1.8G | 76.9 | 93.4 | 
| FreqFormer-XS |  22M | 4.0G | 80.2 | 94.7 |
| FreqFormer-S |  32M | 6.6G | 81.7 | 95.6 | 
| FreqFormer-B |  57M | 11.5G | 82.1 | 95.7 | 





### Requirements

- python =3.8 (conda create -y --name freqformer python=3.8)
- torch>=1.10.0 (conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch)
- timm (pip install timm)


**Data preparation**: download and extract ImageNet images from http://image-net.org/. The directory structure should be

```
│ILSVRC2012/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### Evaluation

To evaluate a pre-trained freqformer model on the ImageNet validation set with a single GPU, run:

```
python infer.py --data-path /path/to/ILSVRC2012/ --arch arch_name --model-path /path/to/model
```


### Training

#### ImageNet

To train freqformer models on ImageNet from scratch, run:

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main_freqformer.py  --output_dir logs/freqformer-xs --arch freqformer-xs --batch-size 128 --data-path /path/to/ILSVRC2012/
```

To finetune a pre-trained model at higher resolution, run:

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main_freqformer.py  --output_dir logs/freqformer-xs-img384 --arch freqformer-xs --input-size 384 --batch-size 64 --data-path /path/to/ILSVRC2012/ --lr 5e-6 --weight-decay 1e-8 --min-lr 5e-6 --epochs 30 --finetune /path/to/model
```

#### Transfer Learning Datasets

To finetune a pre-trained model on a transfer learning dataset, run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main_freqformer_transfer.py  --output_dir logs/freqformer-xs-cars --arch freqformer-xs --batch-size 64 --data-set CARS --data-path /path/to/stanford_cars --epochs 1000 --lr 0.0001 --weight-decay 1e-4 --clip-grad 1 --warmup-epochs 5 --finetune /path/to/model 
```

