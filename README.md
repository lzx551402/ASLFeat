# ASLFeat implementation

![Framework](imgs/framework.png)

TensorFlow implementation of ASLFeat for CVPR'20 paper ["ASLFeat: Learning Local Features of Accurate Shape and Localization"](https://arxiv.org/abs/1904.04084), by Zixin Luo, Lei Zhou, Xuyang Bai, Hongkai Chen, Jiahui Zhang, Yao Yao, Shiwei Li, Tian Fang and Long Quan.

This paper presents a joint learning framework of local feature detectors and descriptors. Two aspects are addressed to learn a powerful feature: 1) shape-awareness of feature points, and 2) the localization accuracy of keypoints. If you find this project useful, please cite:

```
@article{luo2020aslfeat,
  title={ASLFeat: Learning Local Features of Accurate Shape and Localization},
  author={Luo, Zixin and Zhou, Lei and Bai, Xuyang and Chen, Hongkai and Zhang, Jiahui and Yao, Yao and Li, Shiwei and Fang, Tian and Quan, Long},
  journal={Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```

## Requirements

Please use Python 3.7, install NumPy, OpenCV (3.4.2), OpenCV-Contrib (3.4.2) and TensorFlow (1.15.2). Refer to [requirements.txt](requirements.txt) for some other dependencies.

If you are using conda, you may configure ASLFeat as:

```bash
conda create --name aslfeat python=3.7 && \
pip install -r requirements.txt && \
conda activate aslfeat
```

## Get started

Clone the repo and download the pretrained model:
```bash
git clone https://github.com/lzx551402/aslfeat.git && \
cd ASLFeat/pretrained && \
wget https://research.altizure.com/data/aslfeat_models/aslfeat.tar && \
tar -xvf aslfeat.tar
```

A quick example for image matching can be called by:

```bash
cd /local/aslfeat && python image_matching.py --config configs/matching_eval.yaml
```

You may configure ``configs/matching_eval.yaml`` to test images of your own.

## Evaluation scripts

### 1. Benchmark on [HPatches dataset](http://icvl.ee.ic.ac.uk/vbalnt/hpatches)

TODO

### 2. Benchmark on [FM-Bench](http://jwbian.net/fm-bench)

Download the (customized) evaluation pipeline, and follow the instruction to download the [testing data](https://1drv.ms/f/s!AiV6XqkxJHE2g3ZC4zYYR05eEY_m):
```bash
git clone https://github.com/lzx551402/FM-Bench.git
```

Configure ``configs/fmbench_eval.yaml`` and call:

```bash
cd /local/aslfeat && python evaluations.py --config configs/fmbench_eval.yaml
```

The extracted features will be stored in ``FM-Bench/Features_aslfeat``. Use Matlab to run ``Pipeline/Pipeline_Demo.m"`` then ``Evaluation/Evaluate.m`` to obtain the results.

### 3. Benchmark on [visual localization](https://www.visuallocalization.net/)

Download the [Aachen Day-Night dataset](https://www.visuallocalization.net/datasets/) and follow the [instructions](https://github.com/tsattler/visuallocalizationbenchmark) to configure the evaluation.

Configure ``data_root`` in ``configs/aachen_eval.yaml``, and call:

```bash
cd /local/aslfeat && python evaluations.py --config configs/aachen_eval.yaml
```

The extracted features will be saved alongside their corresponding images, e.g., the features for image ``/local/Aachen_Day-Night/images/images_upright/db/1000.jpg`` will be in the file ``/local/Aachen_Day-Night/images/image_upright/db/1000.jpg.aslfeat_ms`` (the method name here is ``aslfeat_ms``).

Finally, refer to the [evaluation script](https://github.com/tsattler/visuallocalizationbenchmark/blob/master/local_feature_evaluation/reconstruction_pipeline.py) to generate and submit the results to the challenge website.

### 4. Benchmark on [Oxford Buildings dataset](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) for image retrieval

Take [Oxford Buildings dataset](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) as an example. First, download the evaluation data and (parsed) groundtruth files:

```bash
mkdir Oxford5k && \
cd Oxford5k && \
mkdir images && \
wget https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz && \
tar -xvf oxbuild_images.tgz -C images && \
wget https://research.altizure.com/data/aslfeat_models/oxford5k_gt_files.tar && \
tar -xvf ... 
```

This script also allows for evaluating [Paris dataset](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/). The (parsed) groundtruth files can be found [here](https://research.altizure.com/data/aslfeat_models/paris6k_gt_files.tar). Be noted to delete the [corrupted images](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/corrupt.txt) of the dataset, and put the remaining images under the same folder.

Next, configure ``configs/oxford_eval.yaml``, and extract the features by:

```bash
cd /local/aslfeat && python evaluations.py --config configs/oxford_eval.yaml
```

We use Bag-of-Words (BoW) method for image retrieval. To do so, clone and compile [libvot](https://github.com/hlzz/libvot.git):

```bash
cd Oxford5k && \
git clone https://github.com/hlzz/libvot.git && \
mkdir build && \
cd build && \
cmake -DLIBVOT_BUILD_TESTS=OFF -DLIBVOT_USE_OPENCV=OFF .. && \
make
```

and the mAP can be obtained by:

```bash
cd Oxford5k && \
python benchmark.py --method_name aslfeat_ms
```

Please cite [libvot](https://github.com/hlzz/libvot.git) if you find it useful.

### 5. Benchmark on [ETH dataset](https://github.com/ahojnnes/local-feature-evaluation)

TODO

### 6. Benchmark on [IMW2020](https://vision.uvic.ca/image-matching-challenge/) 

Download the data (validation/test) [Link](https://vision.uvic.ca/imw-challenge/index.md), then configure ``configs/imw2020_eval.yaml``, finally call:

```bash
cd /local/aslfeat && python evaluations.py --config configs/imw2020_eval.yaml
```