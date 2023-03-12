# BlockCopy - Pedestrian detection with Pedestron

This codebase for pedestrian detection is based on [Pedestron](https://github.com/hasanirtiza/Pedestron).
We removed architectures and datasets that are not useful for our experiments for clarity.

## Installation

Note: a Dockerfile (contributed by a user, thanks!) is available in `./docker/Dockerfile`

**Requirements**

We tested with 
* Pytorch 1.9.1
* CUDA 11.4
* CuPy 9.5
* GCC 9.3
* mmdet 0.6.0

**Installation**

* Install BlockCopy, its requirements and the cityscapes dataset as specified in the [BlockCopy readme](../README.md)
* Install cython: `pip install cython`
* Install pedestron: `python setup.py develop` (if CUDA extensions do not build, check your CUDA setup and if the correct paths are set)


**Dataset**

Prepare Cityscapes video data as in the main [BlockCopy readme](../README.md) and update the `img_root` in the config files located in

    configs/elephant/cityperson


**Model checkpoints**

Get the CSP model from Pedestron from [Google drive](https://drive.google.com/file/d/14qpoyQWIirzUyLZHTxjZe-09AxiUtIxK/view?usp=sharing).
Place in `./checkpoints/` dir resulting in the following structure:
    ./checkpoints/csp/epoch_72.pth

Note that you have to rename the file (remove .stu extension)



## Testing

### Dynamic BlockCopy model

    python ./tools/test_city_person.py configs/elephant/cityperson/csp_r50_clip_blockcopy_030.py ./checkpoints/csp/epoch_ 72 73 --out results/csp_blockcopy_t030.json --num-clips-warmup 400 --num-clips-eval -1

Resulting in 

    Computational cost (avg per img): 380.097 GMACs over 10000 images
    ======= FLOPSCOUNTER =======
    batches: 10000
    # depth 0: 
    model                (MMDataParallel):      380.1 GMac
    # depth 1: 
    module               (CSPBlockCopy):      380.1 GMac
    # depth 2: 
    backbone             (ResNet    ):      85.71 GMac
    bbox_head            (CSPHead   ):     231.09 GMac
    neck                 (CSPNeck   ):      56.79 GMac
    policy               (PolicyTrainRL):       6.51 GMac

    Checkpoint 72: [Reasonable: 11.44%], [Reasonable_Small: 15.31%], [Heavy: 40.56%], [All: 37.47%]


With visualisations of detections, executed blocks, information gain (written to `output/csp_blockcopy_t030`):

    python ./tools/test_city_person.py configs/elephant/cityperson/csp_r50_clip_blockcopy_030.py ./checkpoints/csp/epoch_ 72 73 --out results/csp_blockcopy_t030.json  --save_img --save_img_dir output/csp_blockcopy_t030 --num-clips-warmup 400 --num-clips-eval -1

### Static standard model
    
    python ./tools/test_city_person.py configs/elephant/cityperson/csp_r50_clip.py ./checkpoints/csp/epoch_ 72 73 --out results/csp_blockcopy_t050.json --num-clips-warmup 400 --num-clips-eval -1

