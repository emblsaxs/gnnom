import pytest
from gnnom.mysaxsdocument import saxsdocument
#import filecmp
import os

@pytest.mark.saxsdocument
def test_saxsdocument():
  cur, prop = saxsdocument.read("testdata/SASDA92.dat")
  saxsdocument.write("testdata/SASDA92-rep.dat", {'s': cur['s'], 'I': cur['I'], 'Err': cur['Err']}, prop)
  #assert filecmp.cmp('testdata/SASDA92.dat', 'testdata/SASDA92-rep.dat')
  assert os.path.exists('testdata/SASDA92-rep.dat')
