import os 
import cv2 
import numpy as np
from copy import deepcopy

class DetectionAnnotation(object): 
  def __init__(self): 
    self.boundingboxes = np.zeros((0, 4), np.int32) 
    self.scores = np.zeros((0, 1), np.float32) 
    self.classnames = np.zeros((0, 1), np.dtype('S3'))
    self.im_mat = None
    self.im_path = None

  def set(self, boundingboxes, scores, classnames, im_mat=None, im_path=None):
    assert (im_mat is None and os.path.isfile(im_path)) or \
           (im_path is None and not im_mat is None)
    self.boundingboxes = boundingboxes
    self.scores = scores
    self.classnames = classnames
    self.im_mat = im_mat
    self.im_path = im_path

  def isAccessible(self): 
    if (self.im_mat is not None) or os.path.isfile(self.im_path):
      return True
    else:
      return False

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

    im_rows, im_cols, _ = self.im_mat.shape if self.im_mat\
                            else cv2.imread(self.im_path).shape
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
      im_rows, im_cols, _ = self.im_mat.shape if self.im_mat\
                              else cv2.imread(self.im_path).shape
      crop_box = np.array([[0, 0, im_cols, im_rows]], np.float32)

  def genCropResult(self):
    assert not self.im_mat is None
    im = self.im_mat
    im_rows, im_cols, _ = im.shape
    assert im_rows>im_cols
    new_im_rows, new_im_cols = im_cols, im_rows
    L_SIDE, S_SIDE = im_rows, im_cols

    x_bias, y_bias = 0, -(L_SIDE-S_SIDE)/2
    cx1, cy1, cx2, cy2 = [0, (L_SIDE-S_SIDE)/2, S_SIDE, L_SIDE-(L_SIDE-S_SIDE)/2]
    cropped_im = im[cy1: cy2, cx1: cx2, :]
    n_boxes = self.boundingboxes.shape[0]
    cropped_bbox = self.boundingboxes + np.array([[x_bias, y_bias, x_bias, y_bias] for i in range(n_boxes)], dtype = self.boundingboxes.dtype)
    scores = deepcopy(self.scores)
    classnames = deepcopy(self.classnames)

    cropped_anno = DetectionAnnotation()
    cropped_anno.set(cropped_bbox, scores, classnames, im_mat=cropped_im)
    canvas = cropped_anno.paintBoundingBox()
    cv2.imshow('cropped_anno', canvas)

    # pasted_anno
    pasted_im = np.zeros((S_SIDE, L_SIDE, 3), dtype=cropped_im.dtype)
    pasted_im[:, (L_SIDE-S_SIDE)/2: L_SIDE-(L_SIDE-S_SIDE)/2, :]=cropped_anno.im_mat
    cv2.imshow('pasted_im', pasted_im)
    pasted_bbox = cropped_anno.boundingboxes + np.array([[(L_SIDE-S_SIDE)/2, 0, (L_SIDE-S_SIDE)/2, 0] for i in range(n_boxes)], dtype = cropped_anno.boundingboxes.dtype)
    pasted_scores = deepcopy(cropped_anno.scores)
    pasted_classnames = deepcopy(cropped_anno.classnames)

    pasted_anno = DetectionAnnotation()
    pasted_anno.set(pasted_bbox, pasted_scores, pasted_classnames, im_mat=pasted_im)
    cv2.imshow('pasted_anno', pasted_anno.paintBoundingBox())

    cv2.waitKey()



  def genRotatedAnnotation(self, rotate_clockwise=True):
    PLOT_RESULT = True
    assert self.isAccessible()
    im = self.im_mat if self.im_mat else cv2.imread(self.im_path)
    im_rows, im_cols, _ = im.shape

    corner_pts = np.array([[0, 0], [0, im_rows], [im_cols, 0], [im_cols, im_rows]], dtype=np.float32)
    homo_corner_pts = np.hstack((corner_pts, np.ones((4, 1), dtype=np.float32)))
    homo_mat = None
    if rotate_clockwise:
      homo_mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    else:
      homo_mat = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
    homo_transformed_pts = np.transpose( np.dot(homo_mat, np.transpose(homo_corner_pts)) )
    xmin, xmax = np.min(homo_transformed_pts[:, 0]), np.max(homo_transformed_pts[:, 0])
    ymin, ymax = np.min(homo_transformed_pts[:, 1]), np.max(homo_transformed_pts[:, 1])
    x_bias, y_bias = -xmin, -ymin
    cropped_cols, cropped_rows = xmax-xmin, ymax-ymin

    trimmed_homo_mat = homo_mat + np.array([[0, 0, x_bias], [0, 0, y_bias], [0, 0, 0]], np.float32)
    flipped_im = cv2.warpPerspective(im, trimmed_homo_mat, (cropped_cols, cropped_rows))

    n_boxes = self.boundingboxes.shape[0]
    old_bbox_pts = np.asarray(self.boundingboxes.reshape((n_boxes*2, 2)), dtype=np.float32)
    old_bbox_pts_homo = np.hstack((old_bbox_pts, np.ones((n_boxes*2, 1), dtype=np.float32)))
    new_bbox_pts_homo = np.transpose( np.dot(trimmed_homo_mat, np.transpose(old_bbox_pts_homo)) )
    new_bbox_pts = new_bbox_pts_homo[..., 0:2]
    new_boundingboxes = np.array( new_bbox_pts.reshape((n_boxes, 4)), dtype=np.int32 )

    new_anno_obj = DetectionAnnotation()
    new_anno_obj.set(new_boundingboxes, self.scores, self.classnames, flipped_im)
    new_canvas = new_anno_obj.paintBoundingBox()
    
    if PLOT_RESULT:
      cv2.imshow('im', im)
      cv2.imshow('flip', flipped_im)
      cv2.imshow('canvas', new_canvas)
      cv2.waitKey()
    return new_anno_obj


  def paintBoundingBox(self):
    assert self.isAccessible()
    im = self.im_mat if (not self.im_mat is None) else cv2.imread(self.im_path)
    im_rows, im_cols, _ = im.shape
    canvas_rows, canvas_cols = im_rows, im_cols
    x_bias, y_bias = 0, 0
    if self.boundingboxes.shape[0]>0:
      xmin = np.min(self.boundingboxes[:, 0])
      xmax = np.max(self.boundingboxes[:, 2])
      ymin = np.min(self.boundingboxes[:, 1])
      ymax = np.max(self.boundingboxes[:, 3])
      if xmin<0: x_bias=xmin
      if ymin<0: y_bias=ymin
      if xmax>im_cols: 
        canvas_cols = xmax - xmin
      if ymax>im_rows:
        canvas_rows = ymax - ymin
    canvas = np.zeros((canvas_rows, canvas_cols, 3), dtype=im.dtype)
    canvas[-y_bias: -y_bias+im_rows, -x_bias: -x_bias+im_cols, :] = im
    for i in range(self.boundingboxes.shape[0]):
      x1, y1, x2, y2 = self.boundingboxes[i]
      cv2.rectangle(canvas, (x1-x_bias, y1-y_bias), (x2-x_bias, y2-y_bias), (0, 255, 255), 2)
    return canvas

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
