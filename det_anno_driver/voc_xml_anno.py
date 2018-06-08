import numpy as np
from det_anno_base import DetectionAnnotation
import xml.etree.ElementTree as ET


class VocXmlDetectAnnotation(DetectionAnnotation):
  def __init__(self):
    super(VocXmlDetectAnnotation, self).__init__()

  def loads(self):
    raise NotImplementedError()

  def load(self, xml_path, im_folder):
    with open(xml_path) as in_file:
      tree=ET.parse(in_file)
      root = tree.getroot()
      im_path = str(root.find('filename').text)
      self.im_path = im_path
      imsize = root.find('size')
      im_w = int(imsize.find('width').text)
      im_h = int(imsize.find('height').text)
      all = list()
      for obj in root.iter('object'):
        current = list()
        name = obj.find('name').text
        self.classnames = np.vstack((self.classnames, np.array([name], dtype=np.dtype('S3'))))
        xmlbox = obj.find('bndbox')
        x1 = int(round(float(xmlbox.find('xmin').text)))
        x2 = int(round(float(xmlbox.find('xmax').text)))
        y1 = int(round(float(xmlbox.find('ymin').text)))
        y2 = int(round(float(xmlbox.find('ymax').text)))
        self.boundingboxes = np.vstack((self.boundingboxes, np.array([x1, y1, x2, y2], dtype=np.int32)))
        self.scores = np.vstack(self.scores, np.array([1.], np.float32))

    print self.classnames

  def dump(self):
    raise NotImplementedError()


def test_voc_init():
  v = VocXmlDetectAnnotation()
  print v.impath

def test_voc_load():
  v = VocXmlDetectAnnotation()
  v.load('data/voc-xml/000004.xml', 'data/voc-xml/000004.jpg')

if __name__=='__main__':
  # test_voc_init()
  test_voc_load()
