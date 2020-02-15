# ASLFeat implementation

## Pre-trained model

| Name            | Downloads                                                                         | Descriptions                                                                                                                                                                                                                                                               |
|-----------------|-----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ASLFeat | [Link](https://research.altizure.com/data/aslfeat_models/aslfeat.tar)     | Base ASLFeat model |

The TensorFlow network definition can be found [here](models/cnn_wrapper).

## Example scripts

### 1. Test image matching

To get started, clone the repo and download the pretrained model:
```bash
git clone https://github.com/lzx551402/aslfeat.git && \
cd /local/aslfeat/pretrained && \
wget https://research.altizure.com/data/aslfeat_models/aslfeat.tar && \
tar -xvf aslfeat.tar
```

then call:

```bash
cd /local/aslfeat && python image_matching.py --config configs/matching_eval.yaml
```

### 2. Benchmark on FMBench

Download the data (validation/test) from [here](https://vision.uvic.ca/imw-challenge/index.md), then configure ``configs/imw2020_eval.yaml``, finally call:

```bash
cd /local/aslfeat && python evaluations.py --config configs/imw2020_eval.yaml
```