FCN for Amazon Picking Challenge 2015
=====================================


Usage
-----

.. code-block:: bash

  ./download_dataset.py
  ./train_fcn32s.py


Dataset
-------

- https://drive.google.com/open?id=0B9P1L--7Wd2vbXFvRGJLdy11anM


Result
------

Learning Curve
++++++++++++++


.. image:: static/2016-06-11-23-47-29_log.png


Evaluation
++++++++++

**Condition**

- 2016-06-11-23-47-29: https://drive.google.com/open?id=0B9P1L--7Wd2vWXN4RHNrbXhxZWc
- 7000 iterations

- Accuracy: 82.85 %
- Class accuracy: 53.56 %
- Mean IU: 0.1694
- FWAVACC: 0.7730


**Sample**

.. image:: static/rbo_val_8000/2015-05-06_18-13-09-840861_bin_J.jpg
.. image:: static/rbo_val_8000/2015-05-07_20-26-01-779896_bin_F.jpg
