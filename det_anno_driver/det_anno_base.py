import os 
import cv2 
import numpy as np


class DetectionAnnotation(object): 
  def __init__(self): 
    self.boundingboxes = np.zeros((0, 4), np.int32) 
    self.scores = np.zeros((0, 1), np.float32) 
    self.classnames = np.zeros((0, 1), np.dtype('S3'))
    self.im_path = '' 

  def isHealthy(self): 
    self.health_report = '' 
    is_healthy = True 
    is_curable = True
    if os.path.isfile(self.impath): 
      self.health_report = '# image path location: [PASS]' 
      self.is_curable = False
    else:
      is_healthy = False
      self.health_report = '# image path location: [FAILED]'
    # has same amount 
    # boundingbox are in the image's range

  def paintBBox(self):
    assert self.is_healthy
    im = cv2.imread(self.impath)


def test_det_anno_base_init():
  d = DetectionAnnotation()
  print d.impath
  print d.boundingboxes


if __name__=='__main__':
  test_det_anno_base_init()
