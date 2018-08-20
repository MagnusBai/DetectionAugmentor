import sys
sys.path.insert(0, './')

from DetectionAugmentor.base import DetectionAnnotation

def test_get_iou():
  box1 = np.array([[30, 31, 60, 100]], dtype=np.int32)
  box2 = np.array([[40, 40, 50, 50]], dtype=np.int32)
  box3 = np.array([[30, 40, 45, 60]], dtype=np.int32)
  print "box1 area:", DetectionAnnotation.getBoxArea(box1)
  print "box1 & box2:", DetectionAnnotation.get2BoxIntersectionArea(box1, box2)
  print "box2 & box3:", DetectionAnnotation.get2BoxIntersectionArea(box2, box3)
  print "IoU box1&box2:", DetectionAnnotation.get2BoxIoU(box1, box2)
  print "IoU box2&box3:", DetectionAnnotation.get2BoxIoU(box2, box3)

def test_det_anno_base_init():
  d = DetectionAnnotation()
  print d.impath
  print d.boundingboxes

if __name__=='__main__':
  pass