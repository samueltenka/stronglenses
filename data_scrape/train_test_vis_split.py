''' author: daniel zhang
    date: 2016-11-19
    descr: Prepare dataset by converting to tensorflow indexing and splitting train/test/vis(ualize) 
'''
import numpy
from sklearn.cross_validation import train_test_split
from utils.config import config

X = numpy.load(config["X_PATH_FULL"])
Y = numpy.load(config["Y_PATH_FULL"])

X = [x.transpose((1, 2, 0)) for x in X] # channels-initial to channels-final (for tensorflow)

X_utility, X_vis, Y_utility, Y_vis = train_test_split(X, Y, test_size=0.01, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(X_utility, Y_utility, test_size = 0.1, random_state=30)

numpy.save(config["X_PATH_TRAIN"], X_train)
numpy.save(config["Y_PATH_TRAIN"], Y_train)

numpy.save(config["X_PATH_TEST"], X_test)
numpy.save(config["Y_PATH_TEST"], Y_test)

numpy.save(config["X_PATH_VIS"], X_vis)
numpy.save(config["Y_PATH_VIS"], Y_vis)
