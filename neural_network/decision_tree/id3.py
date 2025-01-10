from __future__ import print_function
import pandas as pd
import numpy as np


class TreeNode(object):
    def __init__(self, ids=None, children=[], entropy=0, depth=0):
        self.ids = ids
        self.entropy = entropy
        self.depth = depth
        self.split_attribute = None
        self.children = children
        self.order = None
        self.label = None