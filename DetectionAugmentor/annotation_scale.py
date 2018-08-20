import numpy as np

class BoundingBoxScale(object):
  def __init__(self, lower_limit, upper_limit, n_level):
    assert upper_limit>lower_limit
    assert n_level>=3 and n_level%2==1
    self.n_level = n_level
    marks = np.linspace(float(lower_limit), \
                        float(upper_limit),
                        n_level+1)
    self.scale_mat = np.zeros((n_level, 2), dtype=float)
    self.scale_mat[:, 0] = marks[:-1]
    self.scale_mat[:, 1] = marks[1:]
    self.nick_name, self.nick_index = [], []
    self._nick_name_id, self._nick_index_id = dict(), dict()
    self.updateIndexInfo()

  def updateIndexInfo(self):
    mid_index = self.n_level/2
    for i in range(self.n_level):
      self.nick_index.append(i-mid_index)
      if i<mid_index:
        self.nick_name.append('X'*(mid_index-i-1)+'S')
      elif i==mid_index:
        self.nick_name.append('M')
      elif i>mid_index:
        self.nick_name.append('X'*(i-mid_index-1)+'L')
    # print self.nick_name, self.nick_index
    self._nick_name_id = dict( zip(self.nick_name, range(self.n_level)) )
    self._nick_index_id = dict( zip(self.nick_index, range(self.n_level)) )
    # print self._nick_name_id, self._nick_index_id

  def getBoundingBoxSizeInfo(self, bb_mat):
    assert isinstance(bb_mat, np.ndarray)
    

if __name__=='__main__':
  pass
