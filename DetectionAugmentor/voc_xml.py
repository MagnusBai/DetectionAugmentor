import numpy as np
from base import DetectionAnnotation
import xml.etree.ElementTree as ET
import os
from lxml.etree import Element, SubElement, tostring

class VocXmlDetectAnnotation(DetectionAnnotation):
  def __init__(self, xml_path, im_folder):
    super(VocXmlDetectAnnotation, self).__init__()
    self.load(xml_path, im_folder)

  def loads(self):
    raise NotImplementedError()

  def load(self, xml_path, im_folder):
    with open(xml_path) as in_file:
      tree=ET.parse(in_file)
      root = tree.getroot()
      im_name = str(root.find('filename').text)   # just support string name
      self.im_path = os.path.join(im_folder, im_name)
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
        self.scores = np.vstack((self.scores, np.array([1.], np.float32)))
    # print self.classnames

  def dumps(self):
    raise NotImplementedError()

  def dump(self, xml_path, im_path):
    # raise NotImplementedError()
    assert self.isAccessible()
    im_folder, im_filename = os.path.split(im_path)
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = '{}'.format(im_folder)
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = '{}'.format(im_path)


if __name__=='__main__':
  pass
