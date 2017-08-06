# FCN for VOC class segmentation


## Inference

```bash
./infer.py -i <IMAGE_FILE> -o /tmp
```


## Training

```bash
./download_datasets.py
./download_models.py

./train_fcn32s.py --gpu 0
./train_fcn16s.py --gpu 0
./train_fcn8s.py --gpu 0
./train_fcn8s_atonce.py --gpu 0
```


## Evaluation

```bash
./evaluate.py ~/data/models/chainer/fcn32s_from_caffe.npz
./evaluate.py ~/data/models/chainer/fcn16s_from_caffe.npz
./evaluate.py ~/data/models/chainer/fcn8s_from_caffe.npz
./evaluate.py ~/data/models/chainer/fcn8s-atonce_from_caffe.npz
```


## Caffe to Chainer model

Firstly, you need to install caffe with pycaffe.

```bash
./caffe_to_chainermodel.py
```
