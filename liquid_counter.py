import numpy as np

class LiquidCounter:
    def __init__(self):
        self.reading = 0
        self.last_five = np.zeros(5)
    
    def update(self, new_reading):
        change = new_reading - self.reading
        #if absolute change more than 10% of reading
        # if abs(change) > 0.1 * self.reading:
        #     return
        self.last_five[:-1] = self.last_five[1:]
        self.last_five[-1] = new_reading
        self.reading = np.mean(self.last_five)
        return