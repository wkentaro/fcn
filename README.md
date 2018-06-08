fcn - Fully Convolutional Networks
==================================

[![PyPI Version](https://img.shields.io/pypi/v/fcn.svg)](https://pypi.python.org/pypi/fcn)
[![Python Versions](https://img.shields.io/pypi/pyversions/fcn.svg)](https://pypi.org/project/fcn)
[![Build Status](https://api.travis-ci.org/wkentaro/fcn.svg?branch=master)](https://travis-ci.org/wkentaro/fcn)

Chainer implementation of [Fully Convolutional Networks](https://github.com/shelhamer/fcn.berkeleyvision.org).


Installation
------------

```bash
pip install fcn
```


Inference
---------

Inference is done as below:

```bash
# forwaring of the networks
img_file=https://farm2.staticflickr.com/1522/26471792680_a485afb024_z_d.jpg
fcn_infer.py --img-files $img_file --gpu -1 -o /tmp  # cpu mode
fcn_infer.py --img-files $img_file --gpu 0 -o /tmp   # gpu mode
```

<img src=".readme/fcn8s_26471792680.jpg" width="80%" >

Original Image: <https://www.flickr.com/photos/faceme/26471792680/>


Training
--------

```bash
cd examples/voc
./download_datasets.py
./download_models.py

./train_fcn32s.py --gpu 0
# ./train_fcn16s.py --gpu 0
# ./train_fcn8s.py --gpu 0
# ./train_fcn8s_atonce.py --gpu 0
```

The accuracy of original implementation is computed with (`evaluate.py`) after converting the caffe model to chainer one
using `convert_caffe_to_chainermodel.py`.\
You can download vgg16 model from here: [`vgg16_from_caffe.npz`](https://drive.google.com/open?id=0B9P1L--7Wd2vRy1XYnRSa1hNSW8).

**FCN32s**

| Implementation | Accuracy | Accuracy Class | Mean IU | FWAVACC | Model File |
|:--------------:|:--------:|:--------------:|:-------:|:-------:|:----------:|
| [Original](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn32s) | 90.4810 | 76.4824 | 63.6261 | 83.4580 | [`fcn32s_from_caffe.npz`](https://drive.google.com/uc?id=0B9P1L--7Wd2vTElpa1p3WFNDczQ) |
| Ours (using `vgg16_from_caffe.npz`) | **90.5668** | **76.8740** | **63.8180** | **83.5067** | [`fcn32s_voc_iter00092000.npz`](https://drive.google.com/uc?0B9P1L--7Wd2vRTQzQl8xcUI5Uk0) |

**FCN16s**

| Implementation | Accuracy | Accuracy Class | Mean IU | FWAVACC | Model File |
|:--------------:|:--------:|:--------------:|:-------:|:-------:|:----------:|
| [Original](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn16s) | 90.9971 | **78.0710** | 65.0050 | 84.2614 | [`fcn16s_from_caffe.npz`](https://drive.google.com/uc?id=0B9P1L--7Wd2vcnBiXzZTcG9FU3c) |
| Ours (using `fcn32s_from_caffe.npz`) | 90.9671 | 78.0617 | 65.0911 | 84.2604 | [`fcn16s_voc_using_fcn32s_from_caffe_iter00032000.npz`](https://drive.google.com/uc?id=0B9P1L--7Wd2vNTFyZDlXel9ZZms) |
| Ours (using `fcn32s_voc_iter00092000.npz`) | **91.1009** | 77.2522 | **65.3628** | **84.3675** | [`fcn16s_voc_iter00100000.npz`](https://drive.google.com/uc?id=0B9P1L--7Wd2vZ1ZUYTJhRkZ1WTg) |

**FCN8s**

| Implementation | Accuracy | Accuracy Class | Mean IU | FWAVACC | Model File |
|:--------------:|:--------:|:--------------:|:-------:|:-------:|:----------:|
| [Original](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn8s) | 91.2212 | 77.6146 | 65.5126 | 84.5445 | [`fcn8s_from_caffe.npz`](https://drive.google.com/uc?id=0B9P1L--7Wd2vb0cxV0VhcG1Lb28) |
| Ours (using `fcn16s_from_caffe.npz`) | 91.2513 | 77.1490 | 65.4789 | 84.5460 | [`fcn8s_voc_using_fcn16s_from_caffe_iter00016000.npz`](https://drive.google.com/uc?id=0B9P1L--7Wd2vdVpRN253el9fdzA) |
| Ours (using `fcn16s_voc_iter00100000.npz`) | **91.2608** | **78.1484** | **65.8444** | **84.6447** | [`fcn8s_voc_iter00072000.npz`](https://drive.google.com/uc?id=0B9P1L--7Wd2vWG5MeUEwWmxudU0) |

**FCN8sAtOnce**

| Implementation | Accuracy | Accuracy Class | Mean IU | FWAVACC | Model File |
|:--------------:|:--------:|:--------------:|:-------:|:-------:|:----------:|
| [Original](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn8s-atonce) | **91.1288** | **78.4979** | **65.3998** | **84.4326** | [`fcn8s-atonce_from_caffe.npz`](https://drive.google.com/uc?id=0B9P1L--7Wd2vZ1RJdXotZkNhSEk) |
| Ours (using `vgg16_from_caffe.npz`) | 91.0883 | 77.3528 | 65.3433 | 84.4276 | [`fcn8s-atonce_voc_iter00056000.npz`](https://drive.google.com/uc?id=0B9P1L--7Wd2vcl9STGhJY1J4WUE) |

<img src="examples/voc/.readme/fcn32s_iter00092000.jpg" width="30%" /> <img src="examples/voc/.readme/fcn16s_iter00100000.jpg" width="30%" /> <img src="examples/voc/.readme/fcn8s_iter00072000.jpg" width="30%" />

Left to right, **FCN32s**, **FCN16s** and **FCN8s**, which are fully trained using this repo. See above tables to see the accuracy.


License
-------

See [LICENSE](LICENSE).
