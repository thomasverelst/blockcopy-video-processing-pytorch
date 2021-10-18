# BlockCopy - semantic segmentation

Semantic segmentation with the [SwiftNet](https://openaccess.thecvf.com/content_CVPR_2019/html/Orsic_In_Defense_of_Pre-Trained_ImageNet_Architectures_for_Real-Time_Semantic_Segmentation_CVPR_2019_paper.html) architecture on Cityscapes with BlockCopy to accelerate video processing.

## Installation
Follow the installation instructions in the [root folder](https://github.com/thomasverelst/blockcopy-video-processing-pytorch/README.md)
Follow the dataset instructions in the same doc. Alternatively, the code can run on dummy data.

## Model checkpoints
Semantic segmentation checkpoints can be found here.
**A slightly improved SwiftNet-RN50 checkpoint will be coming soon**.

| network       | static accuracy  | dynamic accuracy  | link                             |
|---------------|------------------|-------------------|----------------------------------|
| SwiftNet-RN18 | 75.6 @ 104 GMACs | 73.5 @ 63.2 GMACs | [https://drive.google.com/file/d/1-06FdQTegy76A0dv4JDvwPNOsJu6QCZw/view?usp=sharing](https://drive.google.com/file/d/1-06FdQTegy76A0dv4JDvwPNOsJu6QCZw/view?usp=sharing) |
| SwiftNet-RN50 | 77.6 @ 210 GMACs | 76.5 @ 121 GMACs | [https://drive.google.com/file/d/1FtiIEhD9tVMPcJwx41itGCEKU3cyyd0e/view?usp=sharing](https://drive.google.com/file/d/1FtiIEhD9tVMPcJwx41itGCEKU3cyyd0e/view?usp=sharing) |

Download the RN50 checkpoint and place it in the `pretrained/` folder
Note that these checkpoints only have a pretrained static segmentation model, the policy is always trained at test time as specified below.



## Testing

### Dynamic model

To test the **accuracy** of a dynamic model and write output/exection visualizations to `output_demo/rn50_t05/`. Then policy is first warmed up on `--num-clips-warmup` clips (default 600), and then evaluated on the validation set.

    bash configs/swiftnet/swiftnet_rl05.sh

resulting in (might fluctuate, execution policy is not deterministic)

    INFO:root:Number of images: 10000
    INFO:root:Mean IoU 76.85
    INFO:root:Computational cost (avg per img): 118.584 GMACs

To test the **speed** of a dynamic model (disabling metrics and some other code)

    bash configs/swiftnet/swiftnet_rl05_speed.sh

Which should give around 17 FPS on a GTX 1080 Ti.

Note that if you have an I/O bottleneck, you can use the `--single-clip-loop` option.

### Static baseline model

**Accuracy**:

    bash configs/swiftnet/swiftnet_static.sh

Resulting in 

    INFO:root:Mean IoU 77.65
    INFO:root:Computational cost (avg per img): 205.841 GMACs
    INFO:root:### Profiler ###

**Speed**:

    bash configs/swiftnet/swiftnet_static_speed.sh

Which should give around 12 FPS on a GTX 1080 Ti.

## Demo Mode

Instead of using the Cityscapes video dataset, you can use the `demo-dir` argument to evaluate on an arbitrary set of images. The images will be processed as a video after sorting the filenames with natural sort (e.g. img001.jpg, img002.jpg ...). This can be used to visualize the execution decisions on the dataset. Set the number of dataloader workers to 0 to ensure the ordering. 

    python test_swiftnet.py --demo-dir /path/to/folder/with/demo/images --network swiftnet_resnet50 --model-checkpoint pretrained/swiftnet_rn50.pth --output-dir demo_rn50_t05 --block-policy rl_semseg --block-target 0.5 --batch-size 1 --half --workers 0
