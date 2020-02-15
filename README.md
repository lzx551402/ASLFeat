# ASLFeat implementation

## Pre-trained model

| Name            | Downloads                                                                         | Descriptions                                                                                                                                                                                                                                                               |
|-----------------|-----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ASLFeat | [Link](https://research.altizure.com/data/aslfeat_models/aslfeat.tar)     | Base ASLFeat model |

The TensorFlow network definition can be found [here](models/cnn_wrapper).

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

Download the data (validation/test) from [here](https://vision.uvic.ca/imw-challenge/index.md), then configure ``configs/imw2020_eval.yaml``, finally call:

```bash
cd /local/aslfeat && python evaluations.py --config configs/imw2020_eval.yaml
```

### 3. Benchmark on FM-Bench

Download  the data from [here](https://onedrive.live.com/?authkey=%21AELjNhhHTl4Rj-Y&id=36712431A95E7A25%21502&cid=36712431A95E7A25), then configure ``configs/fmbench_eval.yaml``, finally call:

```bash
cd /local/aslfeat && python evaluations.py --config configs/fmbench_eval.yaml
```