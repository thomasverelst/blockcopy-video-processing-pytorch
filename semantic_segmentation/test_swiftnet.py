import argparse
import logging
import os
import os.path as osp
import sys
import time

import cmapy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import blockcopy
from blockcopy.utils.profiler import timings
from torch.utils import data

import lib.ext_transforms as et
from lib import datasets, models
from lib.datasets import CityscapesVid
from lib.utils.metrics import StreamSegMetrics

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="BlockCopy Segmentation Demo")
    parser.add_argument("--demo-dir", type=str, 
                        default="", help="image root for demo")
    parser.add_argument("--cityscapes-dir", type=str, 
                        default="path/to/cityscapes", help="image root for cityscapes evaluation on video")
    parser.add_argument("--model-backbone", default="resnet18", type=str, help="")
    parser.add_argument("--model-checkpoint", default="pretrained/swiftnet_rn18.pth",type=str,
                        help="path to pretrained model checkpoint")
    parser.add_argument("--half", action="store_true", help="run at half precision (fp16)")
    parser.add_argument("--output-dir", default="", type=str, help="subdir for output visualizations")
    parser.add_argument("--res", type=int, default=1024, help="smallest image side in pixels (default Cityscapes resolution is 1024x2048")
    parser.add_argument("--num-clips-warmup", type=int, default=500, help="limit number of clips (-1 to use all clips in training set)")
    parser.add_argument("--num-clips-eval",  type=int, default=-1, help="limit number of clips (-1 to use all clips in test set)")
    parser.add_argument("--clip-length",  type=int, default=20, help="Cityscapes clip length (max 20 frames)")
    parser.add_argument("--batch-size", type=int, default=1, help="Test batch size")
    parser.add_argument("--mode",  type=str, default='val', choices=['val','test'], help='evaluation set')
    parser.add_argument("--fast", action="store_true", help="removes unnecessary operations such as metrics, and displays the FPS")
    parser.add_argument("--single-clip-loop", action="store_true", help="loop single clip to mitigate data loading I/O bottleneck")
    parser.add_argument("--workers", type=int, default=6, help="number of dataloader workers")
    parser.add_argument("--timings", type=int, default=0, help="internal profiler timing priority (0 to disable, 5 for general timings, 10 for detailed timings)")
    
    # add blockcopy arguments to argparser
    blockcopy.add_argparser_arguments(parser)
    
    args = parser.parse_args()
    logging.info(f"Arguments: {args}")

    num_classes = 19
    device = "cuda"  # only support single cuda GPU
    torch.backends.cudnn.benchmark = True
    timings.set_level(args.timings)

    ## Dataset
    val_transform = et.ExtCompose([
        et.ExtResize((args.res,args.res*2)),
        et.ExtToTensor(),
        et.ExtNormalize(mean=CityscapesVid.mean, std=CityscapesVid.std),
    ])
    if args.demo_dir:
        has_labels = False
        dataset_warmup = datasets.demo.DemoImageDataset(root=args.demo_dir, transform=val_transform)
        dataloader_warmup = data.DataLoader(dataset_warmup, shuffle=False, batch_size=args.batch_size, num_workers=args.workers, pin_memory=False)
        dataset_eval = datasets.demo.DemoImageDataset(root=args.demo_dir, transform=val_transform)
        dataloader_eval = data.DataLoader(dataset_eval, shuffle=False, batch_size=args.batch_size, num_workers=args.workers, pin_memory=False)
    
    elif args.cityscapes_dir:
        # cityscapes test evaluation
        has_labels = not args.fast and args.mode != 'test'
        dataset_warmup = CityscapesVid(root=args.cityscapes_dir, split='train',
                                        transform=val_transform, clip_length=args.clip_length, has_labels=has_labels)
        dataset_eval = CityscapesVid(root=args.cityscapes_dir, split=args.mode,
                                        transform=val_transform, clip_length=args.clip_length, has_labels=has_labels)
        dataloader_warmup = data.DataLoader(dataset_warmup, shuffle=False, batch_size=args.batch_size, num_workers=args.workers, pin_memory=False)
        dataloader_eval = data.DataLoader(dataset_eval, shuffle=False, batch_size=args.batch_size, num_workers=args.workers, pin_memory=False)
    else:
        raise AttributeError

    ## Model
    backbone = models.swiftnet.backbones.resnet.__dict__[args.model_backbone](pretrained=False)
    model = models.swiftnet.swiftnet.SwiftNet(backbone=backbone, num_classes=num_classes, num_features=128, use_spp=True)

    ## Load model weights
    model_checkpoint_path = args.model_checkpoint
    assert os.path.isfile(model_checkpoint_path), f"Demo requires model checkpoint!: File not found: {model_checkpoint_path}"
    logging.info(f"=> loading model checkpoint '{model_checkpoint_path}'")
    checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    best_acc = checkpoint["best_acc"] if "best_acc" in checkpoint else "unknown"
    epoch = checkpoint["epoch"] if "epoch" in checkpoint else "unkonwn"
    logging.info(
        f"=> loaded model checkpoint '{model_checkpoint_path}'\
        (epoch {epoch}, best_acc {best_acc})"
    )


    model.eval()
    
    # BlockCopy policy
    static = args.block_policy == 'static'
    if not static:
        # wrap model with blockcopy
        model = blockcopy.BlockCopyModel(model, settings=vars(args))
    model = model.to(device)

    # fuse BN
    import lib.utils.bn_fusion as bn_fusion
    model = bn_fusion.fuse_bn_recursively(model)
    
    # fp16 if needed
    if args.half:
        # run model in FP16
        model = model.half()
        if not static and model.policy.net is not None:
            # run policy in FP32 for training
            model.policy.net = model.policy.net.float()

    # setup output saving
    if args.output_dir:
        assert not args.fast, "Cannot combine fast option with output_dir"
        output_dir = os.path.join('output_demo', args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None
    
    # define warmup/eval loop
    def process_dataset(dataloader, phase, max_num_clips=-1):
        metrics = StreamSegMetrics(num_classes, classes=CityscapesVid.fine_classes)
        timings.reset()
        dataloader_iter = iter(dataloader)

        if max_num_clips >= 0:
            total_num_clips = min(max_num_clips//args.batch_size, len(dataloader))
        else:
            total_num_clips = len(dataloader)
        
        logging.info(f'#-------------------------------------- {phase} --------------------------------------#')
        logging.info(f'## Processing dataset for phase {phase} with {len(dataloader)} clips limited to {max_num_clips}')
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        num_images = 0

        for iteration in tqdm.tqdm(range(total_num_clips)):
            with timings.env('process_dataset/dataloader', 1):
                if not args.single_clip_loop or iteration == 0:
                    clip, target, meta = next(dataloader_iter)
            
            num_images += len(clip)*clip[0].size(0)
            with timings.env('process_dataset/process_clip', 1):
                final_outputs = process_clip(clip, meta, phase)

            if has_labels:
                with timings.env('process_dataset/metrics_update', 1):
                    metrics.update(target.cpu().numpy(), final_outputs.cpu().numpy())

        torch.cuda.synchronize()
        stop = time.perf_counter()

        logging.info(f'Number of images: {num_images}')
        if phase == 'eval':
            metric_results = metrics.get_results()
            logging.info(f"Mean IoU {metric_results['Mean IoU']*100:.2f}")
            if args.fast:
                logging.info(f'Average FPS: {num_images/(stop - start):.2f}')
            if count_flops:
                logging.info(f'Computational cost (avg per img): {model.compute_average_flops_cost()[0]/1e9:.3f} GMACs')
            logging.info(timings)
            logging.info(f'All sementation metrics: {metric_results}')
        

    def process_clip(clip, clip_meta, phase):
        clip_length = len(clip)
        if not static:
            model.reset_temporal()

        for frame_id in range(clip_length):

            inputs = clip[frame_id]
            timings.add_cnt(cnt=inputs.size(0))    
            assert inputs.dim() == 4, inputs.dim()
            assert inputs.size(1) == 3, inputs.size(1)
            inputs = inputs.to(device, non_blocking=True, dtype=torch.float16 if args.half else torch.float32)

            with timings.env('process_clip/model', 2):
                with torch.no_grad():
                    out = model(inputs)
                if frame_id == clip_length - 1 or output_dir:
                    out = F.interpolate(out, scale_factor=4.0, mode='bilinear')
                    preds = out.detach().max(dim=1)[1].cpu()

            # viz if needed
            if output_dir:
                with timings.env('process_clip/viz', 2):
                    if phase == 'warmup':
                        continue
                    phase_output_dir = os.path.join(output_dir, phase)
                    os.makedirs(phase_output_dir, exist_ok=True)
                    relpath = clip_meta["relpath"][0]
                    fname = '.'.join(relpath.replace('/','-').split('.')[:-1])+f'_{frame_id}'
                    
                    logging.info(f'Writing outputs for {fname} to {phase_output_dir}')
                    rescale_func = lambda x: cv2.resize(x, dsize=(1024, 512), interpolation=cv2.INTER_NEAREST)
                    
                    # input image
                    from lib.utils.misc import denormalize
                    image = inputs[0].float()
                    frame = np.clip(rescale_func(denormalize(image, CityscapesVid.mean, CityscapesVid.std).cpu().numpy().transpose(1,2,0)), a_min=0, a_max=1)
                    plt.imsave(osp.join(phase_output_dir, f'{fname}_input.jpg'), frame)
                    
                    # output
                    preds_color = rescale_func(CityscapesVid.decode_target(preds[0]).astype(np.float32)/255)
                    plt.imsave(osp.join(phase_output_dir, f'{fname}_output.jpg'), preds_color)
                    
                    # exec grid
                    if hasattr(model, 'policy_meta'):
                        grid = model.policy_meta['grid']
                        t = rescale_func(grid[0,0].float().cpu().numpy())
                        t = cv2.cvtColor(t*255, cv2.COLOR_GRAY2BGR).astype(np.uint8)
                        t = cv2.applyColorMap(t, cmapy.cmap('viridis')).astype(np.float32)/255
                        t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
                        t = cv2.addWeighted(frame,0.6,t,0.4,0)
                        plt.imsave(os.path.join(phase_output_dir, f'{fname}_grid.jpg'), t)
        return preds

    ## WARMUP phase
    count_flops = False
    process_dataset(dataloader_warmup, 'warmup', max_num_clips=args.num_clips_warmup)

    ## EVAL phase
    torch.backends.cudnn.benchmark = False
    count_flops = not args.fast
    if count_flops:
        # flops counter
        import ptflops.flops_counter
        ptflops.flops_counter.add_flops_counting_methods(model)
        model.start_flops_count(ost=sys.stderr, verbose=False, ignore_list=[])

    process_dataset(dataloader_eval, 'eval', max_num_clips=args.num_clips_eval)


main()
