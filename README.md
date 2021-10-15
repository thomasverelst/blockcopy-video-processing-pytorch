# BlockCopy 
*High-Resolution Video Processing with Block-Sparse Feature Propagation and Online Policies*

[[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Verelst_BlockCopy_High-Resolution_Video_Processing_With_Block-Sparse_Feature_Propagation_and_Online_ICCV_2021_paper.pdf)

<img src="https://thomasverelst.github.io/teaser.gif" />

<img src="https://thomasverelst.github.io/blockcopy_teaser.png" width="300" />\


## Code structure
There is an installable package of the blockcopy framework in `blockcopy` folder, which is used in the codebases for the separate tasks.
Each task has its own readme with additional instructions for that codebase.

* [Pedestrian detection (Pedestron)](./Pedestron/): integration of BlockCopy in Pedestron for pedestrian detection. More difficult to install as it requires CUDA compilation.
* [Semantic segmentation](./semantic_segmentation/): example implementation of BlockCopy with a semantic segmentation backbone, although 
    semantic segmentation is not an optimal application since every region has changing outputs. Easier to understand and install.

Coming later:

* more documentation in the codebase and implementation
* code improvements for better analysis and debugging
* more efficient code with less overhead
* improved semantic segmentation model

## Installation
This code requires an NVIDIA CUDA-capable GPU (no CPU support), with a recent Pytorch version and CuPy:

Create a new Anaconda env and activate it
    
    conda create -n blockcopy python=3.9 -y
    conda activate blockcopy

Install Pytorch 1.9.1

    conda install pytorch=1.9.1 torchvision  cudatoolkit=11.1 -c pytorch -c nvidia

Install other requirements

    pip install -r requirements.txt

Install cupy (see installation instructions of CuPy in case of installation problems)
    
    pip install cupy-cuda111

Install our blockcopy module

    cd ./blockcopy
    python setup.py develop
    cd ..

If any other packages are missing, they should be easily installable using pip. Note that Pedestron requires extra installation steps.

## Dataset preparation
### Requirements
This code uses the [Cityscapes](https://www.cityscapes-dataset.com/) dataset, with the video frames in the set called `leftImg8bit_sequence_trainvaltest.zip (324GB)` (MD5: 4348961b135d856c1777f7f1098f7266), which you might have to request on the download page. Note that the semantic segmentation codebase has a demo option to work on any image dataset for demo purposes when the dataset is not available.

Required Cityscapes packages:
* `gtFine_trainvaltest.zip (241MB)`
* `leftImg8bit_trainvaltest.zip (11GB)`
* `leftImg8bit_sequence_trainvaltest.zip (324GB)`

### Dataset folder structure

Unpack thsoe in a folder, which should result in the following structure:

    cityscapes/
    cityscapes/leftImg8bit/
    cityscapes/leftImg8bit/test/
    cityscapes/leftImg8bit/test/berlin/...
    cityscapes/leftImg8bit/train/
    cityscapes/leftImg8bit/train/aachen/...
    cityscapes/leftImg8bit/val/
    cityscapes/leftImg8bit/val/...
    cityscapes/leftImg8bit_sequence/
    cityscapes/leftImg8bit_sequence/test/
    cityscapes/leftImg8bit_sequence/test/berlin/berlin_000000_000000_leftImg8bit.jpg
    cityscapes/leftImg8bit_sequence/test/berlin/...
    cityscapes/leftImg8bit_sequence/test/...
    cityscapes/leftImg8bit_sequence/train/
    cityscapes/leftImg8bit_sequence/train/aachen/aachen_000000_000000_leftImg8bit.jpg
    cityscapes/leftImg8bit_sequence/train/....
    cityscapes/leftImg8bit_sequence/val/
    cityscapes/leftImg8bit_sequence/val/...
    cityscapes/gtFine/
    cityscapes/gtFine/val/
    cityscapes/gtFine/train/
    cityscapes/gtFine/test/

