"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.

Next, run the analysis-scripts/extract-npz.py (if save_intermediate_xstarts = False) or extract-intermediate-npz.py (if save_intermediate_xstarts = True)
Then, run analysis-scripts/eval-metrics.py
"""

import argparse
import os, sys
from PIL import Image
import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from improved_diffusion.topoloss_normal import TopoLossMSE2D

# NEED TO BE FILLED BY YOU
save_intermediate_xstarts = False # Set to True if we want to save x0,eps,xt at all timesteps
save_custom_conditional = True # If you want to condition using topological constraints
my_constraints = [1,2,3,4,5,6,7] # The topological constraints you want to generate samples for
numwant = 2 # 4, 12 etc number of samples per topological constraint; make sure batch_size is divisible
save_obj_cond = False # Set to True if we want to use object condition, eg giraffe in coco

save_two_cls_cond = False # If you want to simultaneously condition on 0-dim and 1-dim ; # not implemented when save_intermediate_xstarts = True
# Define ranges for a and b; where a is for 0-dim and b is for 1-dim; simultaneous conditioning
a_range = [0, 1, 2, 3]
a_b_pairs = [(a, b) for a in a_range for b in range(a + 1)]


def main_intermediate(): # called when save_intermediate_xstarts = True
    args = create_argparser().parse_args()

    args.batch_size = 1 # makes it easier to save intermediate

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []

    if save_custom_conditional:
        want_labels = []
        for n in my_constraints:
            want_labels.extend([n]*numwant)
        if save_obj_cond:
            o_want_labels = []
            temp = want_labels.copy()
            want_labels = []
            for n in range(NUM_CLASSES):
                o_want_labels.extend([n]*(numwant*len(my_constraints)))
                want_labels.extend(temp)
            assert len(want_labels) == len(o_want_labels)
        #print(want_labels)

    cntr = 0
    o_cntr = 0
    while len(all_images) * args.batch_size < args.num_samples:

        cur_idx = len(all_images)

        model_kwargs = {}
        if args.class_cond:
            if save_custom_conditional:
                val = want_labels[cntr]
                classes = th.full(size=(args.batch_size,), fill_value = float(val), device=dist_util.dev())
                cntr += args.batch_size
            else:
                classes = th.randint(
                    low=1, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                )
            if save_obj_cond:
                val = o_want_labels[o_cntr]
                o_classes = th.full(size=(args.batch_size,), fill_value = int(val), device=dist_util.dev())
                o_cntr += args.batch_size
                model_kwargs["y_obj"] = o_classes
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample, intermediate, intermediate_eps, intermediate_x = sample_fn(
            model,
            (args.batch_size, args.num_colors, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            save_intermediate_xstarts = save_intermediate_xstarts
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        for idx, _ in enumerate(intermediate): # x_starts (approx x0s)
            intermediate[idx] = ((intermediate[idx] + 1) * 127.5).clamp(0, 255).to(th.uint8)
            intermediate[idx] = intermediate[idx].permute(0, 2, 3, 1)
            intermediate[idx] = intermediate[idx].contiguous()
            intermediate[idx] = intermediate[idx].cpu().numpy()

        for idx, _ in enumerate(intermediate_eps): # noise epsilons
            intermediate_eps[idx] = ((intermediate_eps[idx] + 1) * 127.5).clamp(0, 255).to(th.uint8)
            intermediate_eps[idx] = intermediate_eps[idx].permute(0, 2, 3, 1)
            intermediate_eps[idx] = intermediate_eps[idx].contiguous()
            intermediate_eps[idx] = intermediate_eps[idx].cpu().numpy()

        for idx, _ in enumerate(intermediate_x): # xt's
            intermediate_x[idx] = ((intermediate_x[idx] + 1) * 127.5).clamp(0, 255).to(th.uint8)
            intermediate_x[idx] = intermediate_x[idx].permute(0, 2, 3, 1)
            intermediate_x[idx] = intermediate_x[idx].contiguous()
            intermediate_x[idx] = intermediate_x[idx].cpu().numpy()

        # save intermediate x_starts (approx x0's) sample_t_label.npy
        itmd_arr = np.concatenate(intermediate, axis=0)
        if args.class_cond:

            savename = "xstart_{}_{}label_{}timesteps.npz".format(str(cur_idx).zfill(3), classes.cpu().numpy()[0], len(intermediate))
        else:
            savename = "xstart_{}_{}timesteps.npz".format(str(cur_idx).zfill(3), len(intermediate))
        savepath = os.path.join(logger.get_dir(), savename)
        logger.log("saving xstart intermediates of {} to {}".format(str(cur_idx).zfill(3), savepath))
        np.savez(savepath, itmd_arr)

        # save intermediates-epsilon (noise output) sample_t_label.npy
        itmd_arr = np.concatenate(intermediate_eps, axis=0)
        if args.class_cond:
            savename = "epsilon_{}_{}label_{}timesteps.npz".format(str(cur_idx).zfill(3), classes.cpu().numpy()[0], len(intermediate_eps))
        else:
            savename = "epsilon_{}_{}timesteps.npz".format(str(cur_idx).zfill(3), len(intermediate_eps))
        savepath = os.path.join(logger.get_dir(), savename)
        logger.log("saving epsilon intermediates of {} to {}".format(str(cur_idx).zfill(3), savepath))
        np.savez(savepath, itmd_arr)

        # save intermediate xt's (iterative denoising) sample_t_label.npy
        itmd_arr = np.concatenate(intermediate_x, axis=0)
        if args.class_cond:
            savename = "xt_{}_{}label_{}timesteps.npz".format(str(cur_idx).zfill(3), classes.cpu().numpy()[0], len(intermediate_x))
        else:
            savename = "xt_{}_{}timesteps.npz".format(str(cur_idx).zfill(3), len(intermediate_x))
        savepath = os.path.join(logger.get_dir(), savename)
        logger.log("saving xt intermediates of {} to {}".format(str(cur_idx).zfill(3), savepath))
        np.savez(savepath, itmd_arr)

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    all_o_labels = []
    all_a_labels = []
    all_b_labels = []

    if save_custom_conditional:
        want_labels = []
        for n in my_constraints:
            want_labels.extend([n]*numwant)
        if save_obj_cond:
            o_want_labels = []
            temp = want_labels.copy()
            want_labels = []
            for n in range(NUM_CLASSES):
                o_want_labels.extend([n]*(numwant*len(my_constraints)))
                want_labels.extend(temp)
            assert len(want_labels) == len(o_want_labels)

    if save_two_cls_cond:
        want_a_b_pairs = []
        for ab in a_b_pairs:
            want_a_b_pairs.extend([ab]*numwant) # = a_b_pairs * numwant  # Repeat for desired samples, but we want same type consecutive, hence explicit loop
        print(f"len a_b_pairs {len(a_b_pairs)}; {a_b_pairs}")
        print(f"len want_a_b_pairs {len(want_a_b_pairs)}; {want_a_b_pairs}")
        print(f"args.num_samples: {args.num_samples}")


    cntr = 0
    o_cntr = 0
    a_b_cntr = 0
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            if save_custom_conditional:
                val = want_labels[cntr]
                classes = th.full(size=(args.batch_size,), fill_value = float(val), device=dist_util.dev())
                cntr += args.batch_size
            else:
                classes = th.randint(
                    low=1, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                )
            if save_obj_cond:
                val = o_want_labels[o_cntr]
                o_classes = th.full(size=(args.batch_size,), fill_value = int(val), device=dist_util.dev())
                o_cntr += args.batch_size
                model_kwargs["y_obj"] = o_classes
            model_kwargs["y"] = classes
        
        
        if save_two_cls_cond:
            a, b = want_a_b_pairs[a_b_cntr]
            a_tensor = th.full(size=(args.batch_size,), fill_value=float(a), device=dist_util.dev())
            b_tensor = th.full(size=(args.batch_size,), fill_value=float(b), device=dist_util.dev())
            a_b_cntr += args.batch_size

            model_kwargs["y0"] = a_tensor
            model_kwargs["y1"] = b_tensor
        
        
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, args.num_colors, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            save_intermediate_xstarts = save_intermediate_xstarts
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

            if args.obj_cond:
                gathered_o_labels = [
                    th.zeros_like(o_classes) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_o_labels, o_classes)
                all_o_labels.extend([o_labels.cpu().numpy() for o_labels in gathered_o_labels])

        if args.two_cls_cond:
            all_a_labels.extend([np.array([a])] * args.batch_size)
            all_b_labels.extend([np.array([b])] * args.batch_size)

        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if args.obj_cond:
        o_label_arr = np.concatenate(all_o_labels, axis=0)
        o_label_arr = o_label_arr[: args.num_samples]
    if args.two_cls_cond:
        o_a_arr = np.concatenate(all_a_labels, axis=0)
        o_a_arr = o_a_arr[: args.num_samples]
        o_b_arr = np.concatenate(all_b_labels, axis=0)
        o_b_arr = o_b_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.obj_cond:
            np.savez(out_path, arr, label_arr, o_label_arr)
        elif args.class_cond:
            np.savez(out_path, arr, label_arr)
        elif args.two_cls_cond:
            np.savez(out_path, arr, o_a_arr, o_b_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")




def create_argparser():
    if save_two_cls_cond:
        defaults = dict(
            clip_denoised=True,
            num_samples=len(a_b_pairs) * numwant,
            batch_size=2, #2, 4, 12 etc
            use_ddim=False,
            model_path="",
        )
    elif save_custom_conditional:
        if save_obj_cond: # count + animal
            defaults = dict(
                clip_denoised=True,
                num_samples=NUM_CLASSES*len(my_constraints)*numwant,
                batch_size=2, #2, 4, 12 etc
                use_ddim=False,
                model_path="",
            )
        else: # count conditioning
            defaults = dict(
                clip_denoised=True,
                num_samples=len(my_constraints)*numwant,
                batch_size=2, #2, 4, 12 etc
                use_ddim=False,
                model_path="",
                topo_scale=0.,
                topo_birth=False,
                topo_death=False,
                topo_noisy=False,
                topo_dim=0,
            )
    else: # unconditional
        defaults = dict(
            clip_denoised=True,
            num_samples=4, #10000, 8, 20, 2
            batch_size=2, #2, 4, 12 etc
            use_ddim=False,
            model_path="",
        )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    if save_intermediate_xstarts:
        main_intermediate()
    else:
        main()
