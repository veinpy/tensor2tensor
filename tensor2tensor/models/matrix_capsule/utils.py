#encoding=utf-8
#__author__=veinpy


class layerIter(object):

    def __init__(self, layerparams):
        self.idx = 0
        self.layerparams = layerparams
        self.n = len(layerparams)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx < self.n:
            val = self.layerparams[self.idx]
            self.idx += 1
            return val
        else:
            raise StopIteration()
