import sys

class MatrixMN(object):

    def __init__(self):
        self.num_rows_ = 0
        self.num_cols_ = 0
        self.values_ = []

    def initialize(self, _m, _n, init = True):
        num_all_old = self.num_rows_ * self.num_cols_
        self.num_rows_ = _m
        self.num_cols_ = _n

        num_all = self.num_rows_ * self.num_cols_

        if num_all_old != num_all: # allocate memory if num_all is changed

            # check if the matrix is too large
            assert(self.num_rows_ * self.num_cols_ <= sys.maxint)
            self.values_ = [None] * num_all

            if init:
                for i in range(0, num_all):
                    self.values_[i] = 0

    def multiply(self, vector, result):
        return None

    def multiplyTransposed(self, vector, result):
        return None

    # Note: You may transpose matrix and then multiply for better performance.
    # See Eigen library. http://eigen.tuxfamily.org/index.php?title=Main_Page

    def cout(self):
        for row in range(0, self.num_rows_):
            for col in range(0, self.num_cols_):
                print self.getValue(row, col),
            print ''

    def get1DIndex(self, row, col):
        assert (row >= 0)
        assert (col >= 0)
        assert (row < self.num_rows_)
        assert (row < self.num_cols_)
        return col + row * self.num_cols_

    def getValue(self, row, col):
        return self.values_[self.get1DIndex(row, col)];