import numpy as np
import pandas as pd
from scipy.ndimage import measurements
import cv2

# 画像を読み込む
image_name = "0"
image = cv2.imread(f"data/floorplan/raw_data/{image_name}.png", cv2.IMREAD_UNCHANGED)

# チャンネルごとのデータを抽出
room_number = image[:, :, 0]  # 1チャンネル目（部屋番号）
room_type = image[:, :, 1]  # 2チャンネル目（部屋のタイプ）
structure = image[:, :, 2]  # 3チャンネル目（構造）
inside_outside = image[:, :, 3]  # 4チャンネル目（内外判定）

# 部屋番号のユニーク値を取得（255や0は無視）
unique_rooms = np.unique(room_number[(room_number > 0) & (inside_outside == 255)])

# print(f"unique_rooms : {unique_rooms}")

# 情報を保存するリスト
room_data = []

# 各部屋ごとに情報を抽出
for room_id in unique_rooms:
    mask = room_number == room_id  # 部屋のマスク
    room_area = np.sum(mask)  # 面積（ピクセル数）
    room_type_id = np.unique(room_type[mask])[0]  # 部屋タイプ（ユニークなので1つ）
    
    # 周囲の長さを計算
    contour = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    perimeter = cv2.arcLength(contour[0], True) if contour else 0
    
    # 重心を計算
    y_center, x_center = measurements.center_of_mass(mask)
    
    # # ドアの接触部分の計算
    # door_mask = (structure == 255) & (inside_outside == 255)  # ドアのマスク
    # adjacent_doors = np.sum(cv2.dilate(mask.astype(np.uint8), np.ones((3, 3)), iterations=1) & door_mask)
    
    # 情報を保存
    room_data.append({
        "Room ID": room_id,
        "Room Type": room_type_id,
        # "Adjacent Doors": adjacent_doors,
        "Perimeter (pixels)": perimeter,
        "Area (pixels)": room_area,
        "Centroid X": x_center,
        "Centroid Y": y_center,
    })

# データフレームに変換してCSV出力
df = pd.DataFrame(room_data)
df.to_csv(f"data/floorplan/input_data/{image_name}.csv", index=False)

print("部屋ごとの情報をCSVに保存しました: room_data.csv")
