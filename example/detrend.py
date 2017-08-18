'''This script will will take reduced LDSS3C data and detrend the data to get results.'''


import sys
sys.path.append('../../detrendersaurus-master/')
from Detrender import Detrender

try:
    d = Detrender(sys.argv[1])
    d.detrend()
except IndexError:
    d = Detrender()
    d.detrend()
    
