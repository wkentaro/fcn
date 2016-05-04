fcn - Fully Convolutional Networks
==================================

.. image:: https://travis-ci.org/wkentaro/fcn.svg?branch=master
    :target: https://travis-ci.org/wkentaro/fcn


This is Chainer_ implementation of fcn.berkeleyvision.org_.

.. _fcn.berkeleyvision.org: https://github.com/shelhamer/fcn.berkeleyvision.org.git
.. _Chainer: https://github.com/pfnet/chainer.git


Features
--------

- Provide FCN8s model for Chainer. [v1.0.0_]
- Copy caffemodel to chainermodel. [v1.0.0_]
- Forwarding with Chainer for pascal dataset. [v1.0.0_]
- Training with Chainer for pascal dataset. [**not yet**]
- Training for APC2015 dataset. [**not yet**]

.. _v1.0.0: https://github.com/wkentaro/fcn/releases/tag/v1.0.0


Installation
------------

.. code-block:: bash

  pip install fcn


For Developers
++++++++++++++

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

  # Download sample image
  wget https://farm2.staticflickr.com/1522/26471792680_a485afb024_z_d.jpg -O sample.jpg

  fcn_forward.py --img-files sample.jpg --gpu -1

.. image:: https://raw.githubusercontent.com/wkentaro/fcn/master/_images/26471792680.jpg

Original Image: https://www.flickr.com/photos/faceme/26471792680/


For Developers
++++++++++++++

.. code-block:: bash

  # This downloads caffemodel and convert it to chainermodel
  ./scripts/caffe_to_chainermodel.py

  # forwarding of the networks
  ./scripts/fcn_forward.py --img-files data/pascal/VOC2012/JPEGImages/2007_000129.jpg

.. image:: https://raw.githubusercontent.com/wkentaro/fcn/master/_images/2007_000129.jpg

Original Image: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/


License
-------
| Copyright (C) 2016 Kentaro Wada
| Released under the MIT license
| http://opensource.org/licenses/mit-license.php
