# [ICLR 2025] TopoDiffusionNet: A Topology-aware Diffusion Model 

## 0) Overview 

This repository contains the implementation for our work "TopoDiffusionNet: A Topology-aware Diffusion Model", accepted to ICLR 2025. **This is the first work to integrate topology with diffusion models.** We propose a loss function $L_{topo}$ to force the diffusion model to generate topologically-faithful images. Given a topological constraint *c*, our model generates images with *c* number of objects or *c* number of holes, depending on the desired topology (0-dim or 1-dim). Our diffusion model generates masks (binary images), which can be later used as condition for [ControlNet](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.pdf).

![Overview](teaser.png?raw=true)

## 1) Relevant links
- arXiv: https://arxiv.org/abs/2410.16646
- openreview: https://openreview.net/forum?id=ZK1LoTo10R

## 2) Code
This work uses the [Ablated Diffusion Model (ADM)](https://proceedings.neurips.cc/paper_files/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf) as the base diffusion model. Hence, the diffusion code is borrowed from https://github.com/openai/improved-diffusion . This code supports training and inference for 3 models: ADM, ADM-T, and TopoDiffusionNet (TDN). Each of these models generate binary images (masks), which can be later used as condition for [ControlNet](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.pdf).
- ADM is an unconditional (uncond) model
- ADM-T takes the topological constraint *c* as condition. In 0-dim, *c* denotes the number of objects, while in 1-dim, *c* denotes the number of holes. ADM-T is trained with the standard $L_{simple}$ loss function.
- TDN is our proposed method which also takes the topological constraint *c* as condition. Additionally, during training, we impose a topology-based loss function $L_{topo}$ to ensure the model generates masks satisfying the constraint. We retain the standard $L_{simple}$ loss during training.

### 2.1) Environment
You would need to replicate the same environment as in https://github.com/openai/improved-diffusion . Please see their repo for more details. To compute $L_{topo}$, we would need homology-computation libraries. Hence, please run the following:
- [Cubical Ripser](https://github.com/shizuo-kaji/CubicalRipser_3dim): `pip install -U cripser`
- [Gudhi](https://gudhi.inria.fr/): `pip install gudhi` 
- In case you get OT missing issue, you can run `pip install POT` or `conda install conda-forge::pot` to resolve it.

### 2.2) Datasets
As mentioned in the paper, we conduct experiments on 4 datasets. For each dataset, we use only the masks (binary images) for training.
- [0-dim] Synthetic shapes dataset: Curated by us. I used a simple python script to generate N random shapes (circle, triangle, rectangle) at any location in the image. If you have difficulty generating such a dataset, I can try releasing it.
- [0-dim] COCO: https://cocodataset.org/#home 
- [1-dim] CREMI: https://cremi.org/data/
- [1-dim] [ROADS](https://www.cs.toronto.edu/~vmnih/data/) or [Google Maps](https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf) -- Both are similar datasets

**Dataset format**: During training, the topological constraint *c* is determined from the image filename. I provide samples from each dataset in dataset-format/ to help understand the following naming convention. 
- For only 0-dim: make sure the image filename has the format `c_xxx.ext`, where c is the number of objects (connected components) in the image, xxx can be any string, and ext is the image extension (png, jpg, jpeg etc)
- For only 1-dim: make sure the image filename has the format `c_xxx.ext`, where c is the number of holes in the image, xxx can be any string, and ext is the image extension (png, jpg, jpeg etc)
- For 0-dim and 1-dim simultaneously: make sure the image filename has the format `u_v_xxx.ext`, where u is the number of objects (connected components) in the image, v is the number of holes in the image, xxx can be any string, and ext is the image extension (png, jpg, jpeg etc) 
- For 0-dim and object type (eg: giraffe, bird etc): make sure the image filename has the format `c_objname_xxx.ext`, where c is the number of objects (connected components) in the image, objname is the type (eg: giraffe, bird etc), xxx can be any string, and ext is the image extension (png, jpg, jpeg etc)

### 2.3) Training procedure
To speed up training, we do the following in sequence:
- Train ADM (uncond) first.
- To train ADM-T, load the checkpoint from ADM (uncond) and then continue training.
- To train TDN, load the checkpoint from ADM-T and then continue training.

### 2.4) Train commands

**2.4.1) ADM (unconditional).** You can load a pre-train model to fine-tune from (eg, I used [lsun_uncond_100M_2400K_bs64.pt](https://openaipublic.blob.core.windows.net/diffusion/march-2021/lsun_uncond_100M_2400K_bs64.pt))
```
MODEL_FLAGS="--class_cond False --num_colors 3 --image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"

DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm False"

TRAIN_FLAGS="--use_topo 0 --lr 2e-5 --batch_size 16 --save_interval 5000 --resume_checkpoint lsun_uncond_100M_2400K_bs64.pt"

export OPENAI_LOGDIR="<fill>"

CUDA_VISIBLE_DEVICES="<fill>" python image_train.py --data_dir <fill> $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

**2.4.2) ADM-T.** Condition on topological constraint *c*. Best to load checkpoint from ADM (uncond) to resume training. You will have to set `load_strict = False` in train_util.py for checkpoint loading to be successful (since we're loading from an unconditional model to a conditional model). Additionally, I have modified the `class_cond` flag from the [original implementation](https://github.com/openai/improved-diffusion) which used nn.Embedding for N classes. My implementation uses a linear layer embedding, and so the number of classes does not need to be specified. The topological constraint *c* can be any number.
```
MODEL_FLAGS="--class_cond True --num_colors 3 --image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"

DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm False"

TRAIN_FLAGS="--use_topo 0 --lr 2e-5 --batch_size 16 --save_interval 2500 --resume_checkpoint <fill>"

export OPENAI_LOGDIR="<fill>"

CUDA_VISIBLE_DEVICES="<fill>" python image_train.py --data_dir <fill> $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

**2.4.3) TDN (Ours)**. Condition on topological constraint *c*, and using $L_{topo}$ during training. Best to load the checkpoint from ADM-T to resume training. If you are enforcing 0-dim topologicial constraint (Shapes, COCO datasets), the argument in `TRAIN_FLAGS` is `--topo_dim 0`. For enforcing 1-dim topological constraint (CREMI, ROADS), the argument in `TRAIN_FLAGS` is `--topo_dim 1`.
```
MODEL_FLAGS="--class_cond True --num_colors 3 --image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"

DIFFUSION_FLAGS="--weight_topo 1e-5 --diffusion_steps 1000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm False"

TRAIN_FLAGS="--use_topo 1 --topo_birth True --topo_death True --topo_noisy False --topo_dim <fill> --lr 2e-5 --batch_size 16 --save_interval 100 --resume_checkpoint <fill>"

export OPENAI_LOGDIR="<fill>"

CUDA_VISIBLE_DEVICES="<fill>" python image_train.py --data_dir <fill> $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

**2.4.4) Variations.** 
- If you would like to additionally condition on the object type (eg: giraffe, zebra etc), add the `-obj_cond True` argument in `MODEL_FLAGS`. The number of object classes (eg: 10 for coco) needs to be set in script_util.py in the `NUM_CLASSES` variable.
- For the topological constraint *c*, we use linear layers to generate the embedding. If you would prefer to use positional embedding encoding (as used for the timestep *t*), you can include the flag `--as_time_enc True` in `MODEL_FLAGS` command. It gives comparable and sometimes better results, and is a possible solution for inference on unseen constraints.
- If you would like to enforce both 0-dim and 1-dim simultaneously, change commands to `MODEL_FLAGS="--two_cls_cond True --num_colors 3 --image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"`, and the loss would be enforced via `TRAIN_FLAGS="--use_topo 3 --topo_birth True --topo_death True --topo_noisy False --lr 2e-5 --batch_size 16 --save_interval 2500 --resume_checkpoint <fill>"`
- Set `--num_colors 1` in `MODEL_FLAGS` if you want to send train on a single-channel image instead of 3 channels.

### 2.5) Inference commands 
To speed up inference, use DDIM instead of DDPM. You can refer to the [original codebase](https://github.com/openai/improved-diffusion) to understand the flags to use for DDIM. 

```
export OPENAI_LOGDIR="<fill>"

CUDA_VISIBLE_DEVICES="<fill>" python image_sample.py --model_path <fill> $MODEL_FLAGS $DIFFUSION_FLAGS

python3 analysis-scripts/extract-npz.py # to extract the images predicted by the above command.

python3 analysis-scripts/eval-metrics.py # to obtain quantitative performance on metrics like Accuracy, F1 etc.
```

Scripts image_sample.py, extract-npz.py, eval-metrics.py etc have comments within them to help understand what variables to set.

### 2.6) Pre-trained weights
To be added

## 3) Applications to Pathology
This work has been extended to generate layouts for pathology image generation (accepted to CVPR 2025). Please check https://github.com/Melon-Xu/TopoCellGen for the paper and implementation.

## 4) Citation
If you found this work useful, please consider citing it as
```
@article{gupta2025topodiffusionnet,
  title={TopoDiffusionNet: A Topology-aware Diffusion Model},
  author={Gupta, Saumya and Samaras, Dimitris and Chen, Chao},
  journal={International Conference on Learning Representations},
  year={2025}
}
```
