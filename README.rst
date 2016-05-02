fcn - Fully Convolutional Networks
==================================

.. image:: https://travis-ci.org/wkentaro/fcn.svg?branch=master
    :target: https://travis-ci.org/wkentaro/fcn


This is Chainer_ implementation of fcn.berkeleyvision.org_.

.. _fcn.berkeleyvision.org: https://github.com/shelhamer/fcn.berkeleyvision.org.git
.. _Chainer: https://github.com/pfnet/chainer.git


Features
--------

- Provide FCN8s model for Chainer. [done]
- Copy caffemodel to chainermodel. [done]
- Forwarding with Chainer for pascal dataset. [done]
- Training with chainer. [not yet]


Installation
------------

.. code-block:: bash

  git clone https://github.com/wkentaro/fcn.git
  cd fcn

  python setup.py install

.. _here: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

You need to download pascal VOC2012 dataset from here_, and install it as below construction::

  - fcn - data - pascal - VOC2012 -- JPEGImages
                                   - SegmentationClass
                                   - ...


Usage
-----

.. code-block:: bash

  # This downloads caffemodel and convert it to chainermodel
  ./scripts/caffe_to_chainermodel.py

  # forwarding of the networks
  ./scripts/forward.py

**Result**

.. image:: _images/2007_000129.jpg

(http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)


License
-------
| Copyright (C) 2016 Kentaro Wada
| Released under the MIT license
| http://opensource.org/licenses/mit-license.php
