import numpy as np
from PIL import Image
import os

get_numpy():
    data_dir = './img/'
    classes = os.listdir(data_dir)

    x = []
    y = []

    for c in classes:
        imgs = os.listdir(data_dir+c)
        for img in imgs:
            arr = np.asarray(Image.open(data_dir+c+'/'+img))
            x.append(arr)
            y.append(c) # use int(c) if required

    x = np.asarray(x)
    y = np.asarray(y)

    print("Done loading")
    print(np.shape(x), np.shape(y))
    return x, y

#get_numpy()
