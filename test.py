import numpy as np
import cv2

# 入力マスク
mask = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0]
], dtype=np.uint8)

# カーネルと膨張処理
kernel = np.ones((3, 3), dtype=np.uint8)
expanded_mask = cv2.dilate(mask, kernel, iterations=1)

print("膨張後のマスク:")
print(expanded_mask)