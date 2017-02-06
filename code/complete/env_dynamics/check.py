import numpy as np

a2_P = np.load('a2_P.npy')
a2_R = np.load('a2_R.npy')
a22_P = np.load('a22_P.npy')
a22_R = np.load('a22_R.npy')
print('check P {}'.format((a2_P==a22_P).all()))
print('check R {}'.format((a2_R==a22_R).all()))
