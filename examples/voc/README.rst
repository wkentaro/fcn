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

.. image:: static/fcn32s_iters.gif


Convert caffemodel to chainermodel
++++++++++++++++++++++++++++++++++

.. code-block:: bash

  ./caffe_to_chainermodel.py
