import os 
import cv2 
import numpy as np


class DetectionAnnotation(object): 
  def __init__(self): 
    self.boundingboxes = np.zeros((0, 4), np.int32) 
    self.scores = np.zeros((0, 1), np.float32) 
    self.classnames = np.zeros((0, 1), np.dtype('S3'))
    self.im_path = ''

  def isAccessible(self): 
    return os.path.isfile(self.im_path) 

  def findBorderAndUnseenBox(self):

    STRICT_BORDER_BOX_IDENTIFICATION=False
    def isStrictBorderBox(obj_box, im_box):
      xobj1, yobj1, xobj2, yobj2 = obj_box[0, :]
      xim1, yim1, xim2, yim2 = im_box[0, :]
      min_margin = 5
      if abs(xim1-xobj1)<min_margin: return True
      if abs(xim2-xobj2)<min_margin: return True
      if abs(yim1-yobj1)<min_margin: return True
      if abs(yim2-yobj2)<min_margin: return True
      return False

    print 'im_path: ', self.im_path
    assert self.isAccessible()
    border_box_flags = np.array([False for i in range(self.boundingboxes.shape[0])])
    unseen_box_flags = np.array([False for i in range(self.boundingboxes.shape[0])])

    im_rows, im_cols, _ = cv2.imread(self.im_path).shape
    im_box = np.array([[0, 0, im_cols, im_rows]], np.float32)
    for i in range(self.boundingboxes.shape[0]):
      obj_box = self.boundingboxes[i, :]
      obj_box = np.expand_dims(obj_box, axis=0)
      obj_box_area = self.getBoxArea(obj_box)
      print obj_box_area, obj_box
      obj_box_in_view_area = self.get2BoxIntersectionArea(obj_box, im_box)
      if obj_box_in_view_area==0:
        unseen_box_flags[i] = True
      elif obj_box_in_view_area<obj_box_area:
        border_box_flags[i] = True
      elif STRICT_BORDER_BOX_IDENTIFICATION and isStrictBorderBox(obj_box, im_box):
        border_box_flags[i] = True
    return border_box_flags, unseen_box_flags

  def genPerspectiveTransformedObj(self, homography_mat, crop_box=None):
    __doc__=''' '''
    new_obj = DetectionAnnotation()
    assert self.isAccessible()
    if crop_box==None: 
      im_rows, im_cols, _ = cv2.imread(self.im_path).shape
      crop_box = np.array([[0, 0, im_cols, im_rows]], np.float32)
    


  def paintBBox(self):
    assert self.is_healthy
    im = cv2.imread(self.impath)

  @staticmethod
  def getBoxArea(box):
    __doc__ = '''box1 box2 is numpy.ndarray() shaped as (1, 4) \n'''
    xmin, ymin, xmax, ymax = box[0, :]
    area = (ymax-ymin)*(xmax-xmin)
    return area

  @staticmethod
  def get2BoxIntersectionArea(box1, box2):
    __doc__ = '''box1 box2 is numpy.ndarray() shaped as (1, 4) \n'''
    xmin1, ymin1, xmax1, ymax1 = box1[0, :]
    xmin2, ymin2, xmax2, ymax2 = box2[0, :]
    if xmin1>=xmax2 or xmin2>=xmax1 or ymin1>=ymax2 or ymin2>=ymax1:
      return 0
    else: 
      _, xinter1, xinter2, _ = sorted([xmin1, xmax1, xmin2, xmax2])
      _, yinter1, yinter2, _ = sorted([ymin1, ymax1, ymin2, ymax2])
      return (xinter2 - xinter1)*(yinter2 - yinter1)

  @staticmethod
  def get2BoxUnionArea(box1, box2):
    __doc__ = '''box1 box2 is numpy.ndarray() shaped as (1, 4) \n'''
    return DetectionAnnotation.getBoxArea(box1) + DetectionAnnotation.getBoxArea(box2)\
            - DetectionAnnotation.getBoxIntersectionArea(box1, box2)

  @staticmethod
  def get2BoxIoU(box1, box2):
    inter_area = DetectionAnnotation.get2BoxIntersectionArea(box1, box2)
    area1, area2 = DetectionAnnotation.getBoxArea(box1), DetectionAnnotation.getBoxArea(box2)
    return float(inter_area)/(area1+area2-inter_area)


def test_det_anno_base_init():
  d = DetectionAnnotation()
  print d.impath
  print d.boundingboxes


def test_get_iou():
  box1 = np.array([[30, 31, 60, 100]], dtype=np.int32)
  box2 = np.array([[40, 40, 50, 50]], dtype=np.int32)
  box3 = np.array([[30, 40, 45, 60]], dtype=np.int32)
  print "box1 area:", DetectionAnnotation.getBoxArea(box1)
  print "box1 & box2:", DetectionAnnotation.get2BoxIntersectionArea(box1, box2)
  print "box2 & box3:", DetectionAnnotation.get2BoxIntersectionArea(box2, box3)
  print "IoU box1&box2:", DetectionAnnotation.get2BoxIoU(box1, box2)
  print "IoU box2&box3:", DetectionAnnotation.get2BoxIoU(box2, box3)


if __name__=='__main__':
  # test_det_anno_base_init()
  test_get_iou()
