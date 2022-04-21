import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


from fitPoint import LinearRegression
if __name__ == '__main__':
    model = LinearRegression()

    model.load_state_dict(torch.load('./checkPoint1.pth'))

    model.eval()


    test_data_list = [[2,2], [2,3], [3,2],[3,3],[4,4]]
    test_data = np.array(test_data_list, dtype=np.float32)
    test_data = torch.from_numpy(test_data)
    value = model(test_data)
    print("value: ", value)
    print('hello')

