# BlockCopy - Pedestrian detection with Pedestron

This codebase for pedestrian detection is based on [Pedestron](https://github.com/hasanirtiza/Pedestron).
We removed architectures and datasets that are not useful for our experiments for clarity.

## Installation

**Requirements**
We tested with 
* Pytorch 1.9.1
* CUDA 11.4
* CuPy 9.5
* GCC 9.3

**Installation **

* Install BlockCopy, its requirements and the cityscapes dataset as specified in the [BlockCopy readme](../README.md)
* Install cython: `pip install cython`
* Install pedestron: `python setup.py develop` (if CUDA extensions do not build, check your CUDA setup and if the correct paths are set)

**Model checkpoints **

Get the CSP model from Pedestron from [Google drive](https://drive.google.com/file/d/14qpoyQWIirzUyLZHTxjZe-09AxiUtIxK/view?usp=sharing).
Place in `./checkpoints/` dir resulting in the following structure:
    ./checkpoints/csp/epoch_72.pth

## Testing

### Dynamic BlockCopy model

    python ./tools/test_city_person.py configs/elephant/cityperson/csp_r50_clip_blockcopy_030.py ./checkpoints/csp/epoch_ 72 73 --out results/csp_blockcopy_t030.json csp_blockcopy_t030 --num-clips-warmup 400 --num-clips-eval -1

Resulting in 

    Total eval images: 10000
    Computational cost (avg per img): 363.129 GMACs over 10000 images
    ======= FLOPSCOUNTER =======
    batches: 10000
    # depth 0: 
    model                (MMDataParallel):     363.13 GMac
    # depth 1: 
    module               (CSPBlockCopy):     363.13 GMac
    # depth 2: 
    backbone             (ResNet    ):      83.22 GMac
    bbox_head            (CSPHead   ):     224.43 GMac
    neck                 (CSPNeck   ):      55.15 GMac
    policy               (PolicyTrainRL):       0.32 GMac

    Average FPS: 7.10 over 10000 images
    Checkpoint 72: [Reasonable: 11.54%], [Reasonable_Small: 15.74%], [Heavy: 40.49%], [All: 37.61%]


With visualisations of detections, executed blocks, information gain (written to `OUTPUT/csp_blockcopy_t030`):

    python ./tools/test_city_person.py configs/elephant/cityperson/csp_r50_clip_blockcopy_030.py ./checkpoints/csp/epoch_ 72 73 --out results/csp_blockcopy_t030.json  --save_img --save_img_dir OUTPUT/csp_blockcopy_t030 --num-clips-warmup 400 --num-clips-eval -1

### Static standard model
    
    python ./tools/test_city_person.py configs/elephant/cityperson/csp_r50_clip.py ./checkpoints/csp/epoch_ 72 73 --out results/csp_blockcopy_t050.json csp_blockcopy_t030 --num-clips-warmup 400 --num-clips-eval -1

