from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
#%matplotlib inline
from caffe2.proto import caffe2_pb2
import numpy as np
import skimage.io
import skimage.transform
from matplotlib import pyplot
import os
from caffe2.python import core, workspace, models
import urllib2
import operator
print("Required modules imported.")
