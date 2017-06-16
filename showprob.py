import numpy
import math
plens = numpy.load('probabilities.npy')
plens = numpy.asarray([x[0] for x in plens])

valid_plens = numpy.asarray([not math.isnan(x) for x in plens])

plens = plens[valid_plens]

ranking = numpy.argsort(plens)[: : -1]
print(ranking)
for i in range(50):
	print(plens[ranking[-i-1]])
