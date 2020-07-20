import numpy as np
class ListOfStrings:
    def __init__(self, listOfStrings):
        self.listOfStrings = listOfStrings
        self.maxlen = np.max(list(map(len, listOfStrings)))