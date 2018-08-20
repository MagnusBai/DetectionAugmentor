import sys
sys.path.insert(0, './')
import os
import numpy as np
from DetectionAugmentor.annotation_scale import BoundingBoxScale


def test_BBScale_init():
  scale = BoundingBoxScale(80, 900, 3)


if __name__=='__main__':
  test_BBScale_init()
