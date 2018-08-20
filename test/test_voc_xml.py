import sys
sys.path.insert(0, './')
import os
import cv2
from DetectionAugmentor.voc_xml import VocXmlDetectAnnotation

def getOutDir():
  out_dir = 'out'
  if not os.path.isdir(out_dir): os.mkdir(out_dir)
  return out_dir
  
def test_voc_init():
  xml_path, im_folder = 'data/voc-xml/000004.xml', 'data/voc-xml/'
  v = VocXmlDetectAnnotation(xml_path, im_folder)
  print dir(v)
  im_painted = v.genPaintedBoxImage()
  cv2.imwrite(os.path.join(getOutDir(), '000004_plot.jpg'), im_painted)

def test_voc_load():
  v = VocXmlDetectAnnotation()
  v.load('data/voc-xml/000004.xml', 'data/voc-xml/000004.jpg')

def test_load_objs():
  v = VocXmlDetectAnnotation('data/voc-xml/000004.xml', 'data/voc-xml/')
  # print v.im_path
  border_flags, unseen_flags = v.findBorderAndUnseenBox()
  print border_flags, unseen_flags

def test_perspective_transform():
  v = VocXmlDetectAnnotation('data/voc-xml/000004.xml', 'data/voc-xml/')

def test_paint_objs():
  v = VocXmlDetectAnnotation('data/voc-xml/000004.xml', 'data/voc-xml/')
  im = v.genPaintedBoxImage()
  cv2.imshow('display', im)
  cv2.waitKey()

def test_flip_anno():
  v = VocXmlDetectAnnotation('data/voc-xml/000004.xml', 'data/voc-xml/')
  v.genRotatedAnnotation(True)

def test_flip_and_crop():
  v = VocXmlDetectAnnotation('data/voc-xml/000004.xml', 'data/voc-xml/')
  v1 = v.genRotatedAnnotation(True)
  v1.genCropResult()

def test_xml_dump():
  v = VocXmlDetectAnnotation('data/voc-xml/000004.xml', 'data/voc-xml/')
  v2 = v.genRotatedAnnotation(True)
  # v2.dump('out/im.xml', 'out/im.jpg')  # TODO: how to get a derived class (VocXmlDetectAnnotation) object, from a based class (DetectionAnnotation) object

def test_remove_border_unseen():
  v = VocXmlDetectAnnotation('data/voc-xml/000004.xml', 'data/voc-xml/')
  v1 = v.genRotatedAnnotation(True)
  v2 = v1.genCropResult()
  v3 = v2.genNoBoarderAndUnseenAnnotation()

if __name__=='__main__':
  test_voc_init()
