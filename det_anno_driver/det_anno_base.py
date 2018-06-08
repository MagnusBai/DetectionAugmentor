import os 
import cv2 
import numpy as np


class DetectionAnnotation(object): 
  def __init__(self): 
    self.boundingboxes = np.zeros((0, 4), np.int32) 
    self.scores = np.zeros((0, 1), np.float32) 
    self.classnames = np.zeros((0, 1), np.dtype('S3'))
    self.im_path = '' 

  def isAccessable(self): 
    return os.path.isfile(self.impath) 

  def hasUnseenBox(self):
    assert self.isAccessable()
    im = cv2.imread(self.im_path)
    rows, cols, _ = im.shape
    x1_unseen = (self.boundingboxes[:, 0] - (cols-1))>=0
    y1_unseen = (self.boundingboxes[:, 1] - (rows-1))>=0

  def paintBBox(self):
    assert self.is_healthy
    im = cv2.imread(self.impath)

  @staticmethod
  def getBoxArea(box):
    __doc__ = '''box1 box2 is numpy.ndarray() shaped as (1, 4) \n'''
    xmin, ymin, xmax, ymax = box[0, :]
    return (ymax-ymin)*(xmax, xmin)

  @staticmethod
  def get2BoxIntersectionArea(box1, box2):
    __doc__ = '''box1 box2 is numpy.ndarray() shaped as (1, 4) \n'''
    xmin1, ymin1, xmax1, ymax1 = box1[0, :]
    xmin2, ymin2, xmax2, ymax2 = box2[0, :]
    if xmin1>=xmax2 or xmin2>=xmin1 or ymin1>=ymax2 or ymin2>=ymax1:
      return 0
    else:
      _, xinter1, xinter2, _ = sorted([xmin1, xmax1, xmin2, xmax2])
      _, yinter1, yinter2, _ = sorted([ymin1, ymax1, ymin2, ymax2])
      return (xinter2 - xinter1)*(yinter2 - yinter1)

  @staticmethod
  def get2BoxUnionArea(box1, box2):
    __doc__ = '''box1 box2 is numpy.ndarray() shaped as (1, 4) \n'''
    return getBoxArea(box1) + getBoxArea(box2) - getBoxIntersectionArea(box1, box2)

  @staticmethod
  def get2BoxIoU(box1, box2):
    inter_area = get2BoxIntersectionArea(box1, box2)
    area1, area2 = getBoxArea(box1), getBoxArea(box2)
    return float(inter_area)/(area1+area2-inter_area)

def test_det_anno_base_init():
  d = DetectionAnnotation()
  print d.impath
  print d.boundingboxes

def test_get_iou():
  box1 = np.array([[30, 51, 60, 40]], dtype=np.int32)
  print getBoxArea(box1)


if __name__=='__main__':
  # test_det_anno_base_init()
  test_get_iou()
