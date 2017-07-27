'''This script will will take reduced LDSS3C data and detrend the data to get results.'''


import sys
sys.path.append('/home/hdiamond/local/lib/python2.7/site-packages/')
sys.path.append('/h/mulan0/code/')
sys.path.append('/h/mulan0/code/mosasaurus')
sys.path.append('/h/mulan0/code/detrendersaurus')
from detrendersaurus.Detrender import Detrender

try:
    d = Detrender(sys.argv[1])
    d.detrend()
except IndexError:
    d = Detrender()
    d.detrend()
    
