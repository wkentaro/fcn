FCN for Pascal VOC
==================


Usage
-----


Training
++++++++

.. code-block:: bash

  ./download_dataset.py
  ./train_fcn32s.py


The learning curve looks like below:

.. image:: static/learning_scale0.8.gif


Convert caffemodel to chainermodel
++++++++++++++++++++++++++++++++++

.. code-block:: bash

  ./caffe_to_chainermodel.py
