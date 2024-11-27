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
n_rooms = len(unique_rooms)

# 情報を保存するリスト
input_data = []
output_data = []

# 隣接行列
door_adjacency_matrix = np.zeros((n_rooms, n_rooms), dtype=int)
wall_adjacency_matrix = np.zeros((n_rooms, n_rooms), dtype=int)
spatial_adjacency_matrix = np.zeros((n_rooms, n_rooms), dtype=int)

# ドア、窓のマスク
wall_mask = (room_type == 16) & (inside_outside == 255)
door_mask = (room_type == 17) & (inside_outside == 255)

# 膨張処理時のカーネル
kernel = np.array([[0,0,0,1,0,0,0],
                   [0,0,0,1,0,0,0],
                   [0,0,0,1,0,0,0],
                   [1,1,1,1,1,1,1],
                   [0,0,0,1,0,0,0],
                   [0,0,0,1,0,0,0],
                   [0,0,0,1,0,0,0]], dtype=np.uint8)

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

    # 膨張範囲
    expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    for room_id_other in unique_rooms:
        if room_id >= room_id_other:
            continue
        mask_other = room_number == room_id_other
        # 相手側の膨張範囲
        expanded_mask_other = cv2.dilate(mask_other.astype(np.uint8), kernel, iterations=1)

        if np.any(expanded_mask & expanded_mask_other & door_mask):
            door_adjacency_matrix[room_id-1, room_id_other-1] = 1
            door_adjacency_matrix[room_id_other-1, room_id-1] = 1
            spatial_adjacency_matrix[room_id-1, room_id_other-1] = 1
            spatial_adjacency_matrix[room_id_other-1, room_id-1] = 1
        elif np.any(expanded_mask & expanded_mask_other & wall_mask):
            wall_adjacency_matrix[room_id-1, room_id_other-1] = 1
            wall_adjacency_matrix[room_id_other-1, room_id-1] = 1
            spatial_adjacency_matrix[room_id-1, room_id_other-1] = 1
            spatial_adjacency_matrix[room_id_other-1, room_id-1] = 1

    # 情報を保存
    input_data.append({
        "Room ID": room_id,
        "Room Type": room_type_id,
        # "Adjacent Doors": adjacent_doors,
        "Perimeter (pixels)": perimeter,
        "Area (pixels)": room_area,
        "Centroid X": x_center,
        "Centroid Y": y_center,
    })

output_data.append({
    "Door Adjacency": door_adjacency_matrix,
    "Wall Adjacency": wall_adjacency_matrix,
    "Spatial Adjacency": spatial_adjacency_matrix,
})

# データフレームに変換してCSV出力
df_input = pd.DataFrame(input_data)
df_output = pd.DataFrame(output_data)
df_input.to_csv(f"data/floorplan/input_data/{image_name}.csv", index=False)
df_output.to_csv(f"data/floorplan/output_data/{image_name}.csv", index=False)

print(f"{image_name}.pngの情報をCSVに保存しました")
