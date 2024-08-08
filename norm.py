import numpy as np
import time


def img2Vector(image: np.array):
    l, h, d = image.shape
    return image.reshape(l, h, d, 1)


print(img2Vector(np.random.rand(3, 3, 2)))


def normaliseRow(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

x = np.array([[0, 1000, 4], [2, 6, 4]])
print(normaliseRow(x))



x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

st = time.process_time()
dot = 0

for i in range(len(x1)):
    dot += x1[i]*x2[i]

sp = time.process_time()

print(f"Time took : {sp-st}")


st = time.process_time()
dot = np.dot(x1, x2)
sp = time.process_time()

print(f"Time took : {sp-st}")
