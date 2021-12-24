<img src="pictures_resnet50_simclr\transistor_28.png" width="700px"></img>

## Anomaly Detection with self-supervised pretrained-model

Implementation of anomaly detection with Simclearv2

## Usage

Simply plugin your neural network, specifying (1) the image dimensions as well as (2) the name (or index) of the hidden layer, whose output is used as the latent representation used for self-supervised training.

```bash
python padim_ssl.py --arch resnet50_simclr
```

## Citation

```bibtex

```