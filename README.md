# ASLFeat implementation

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

## Get started

Clone the repo and download the pretrained model:
```bash
git clone https://github.com/lzx551402/aslfeat.git && \
cd /local/aslfeat/pretrained && \
wget https://research.altizure.com/data/aslfeat_models/aslfeat.tar && \
tar -xvf aslfeat.tar
```

## Example scripts

### 1. Test image matching

Configure ``configs/matching_eval.yaml`` and call:

```bash
cd /local/aslfeat && python image_matching.py --config configs/matching_eval.yaml
```

### 2. Benchmark on IMW2020 

Download the data (validation/test) [Link](https://vision.uvic.ca/imw-challenge/index.md), then configure ``configs/imw2020_eval.yaml``, finally call:

```bash
cd /local/aslfeat && python evaluations.py --config configs/imw2020_eval.yaml
```

### 3. Benchmark on FM-Bench

Download the (revised) evaluation pipeline, and follow the instruction to download the [testing data](https://1drv.ms/f/s!AiV6XqkxJHE2g3ZC4zYYR05eEY_m):
```bash
git clone https://github.com/lzx551402/FM-Bench.git
```

Configure ``configs/fmbench_eval.yaml`` and call:

```bash
cd /local/aslfeat && python evaluations.py --config configs/fmbench_eval.yaml
```

The extracted features will be stored in ``FM-Bench/Features_aslfeat``. Use Matlab to run ``Pipeline/Pipeline_Demo.m"`` then ``Evaluation/Evaluate.m`` to obtain the results.