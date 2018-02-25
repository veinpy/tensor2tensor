#encoding=utf-8
#__author__=veinpy


class layerIter(object):

    def __init__(self, layersizes, ):
        self.idx = 0
        self.n = len(layersizes)

    def __iter__(self):
        return self

    def next(self):
        if self.idx < self.n:
            val = self.idx
            self.idx += 1
            return val
        else:
            raise StopIteration()