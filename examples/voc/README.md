# FCN for VOC class segmentation


## Inference

```bash
./infer.py -i <IMAGE_FILE> -o /tmp
```


## Training

```bash
./download_datasets.py
./train_fcn32s.py --gpu 0
```


## Evaluation

```bash
./download_models.py
./evaluate.py ~/data/models/chainer/fcn8s_from_caffe.npz
./evaluate.py ~/data/models/chainer/fcn16s_from_caffe.npz
./evaluate.py ~/data/models/chainer/fcn32s_from_caffe.npz
```


## Caffe to Chainer model

Firstly, you need to install caffe with pycaffe.

```bash
./caffe_to_chainermodel.py
```
