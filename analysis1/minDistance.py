import json
import svgwrite
import numpy as np
import cairosvg
import fixation_data_utils as f_utils
from tqdm import trange
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import pearsonr
import numpy as np  
import cv2
from scipy.optimize import minimize
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
import colorsys
import random
from scipy.optimize import linear_sum_assignment

important_images = ["DIY_Gantry_bust_010_RGB.png", "PMS_Princeton_penguin1_000_RGBND.png", "WEB_CUHK_cottage1_000_RGB.png"]

with open('./Data/registed_eyetrack_drawing_with_fixation_point.json', 'r') as r_json_file:
    registed_eyetrack_and_drawing_data = json.load(r_json_file)
with open('./Data/eyetrack_drawing_with_fixation_point.json', 'r') as json_file:
    eyetrack_and_drawing_data = json.load(json_file)

num_clusters = 64
centers = [[],[],[]]
less_data = 110000
for uid, value in eyetrack_and_drawing_data.items():
    if uid == "eyetrack09":
        continue
    for image, fixation_datas in value.items():
        if image not in important_images:
            continue
        name = image[:-4] + '_' + uid + '.png'
        data = np.array([np.array([fixation_data["x"], fixation_data["y"], fixation_data["timeSpan"]]) for fixation_data in fixation_datas])
        if len(data) < less_data:
            less_data = len(data)
        # cluster_centers = data[np.random.choice(data.shape[0], num_clusters, replace=False), :2]
        # assignments, cluster_centers = weighted_knn_clustering(data, cluster_centers)
        sorted_data = data[np.argsort(data[:, 2])[::-1]]
        top_xy = sorted_data[:num_clusters, :2]  # 只提取前两列（x 和 y）
        centers[important_images.index(image)].append(top_xy)
        canvas = cv2.imread("Data/select_image/" + image, cv2.IMREAD_COLOR)
        
    print(uid)
ccs = np.zeros((3, 10, 10))
ps = np.zeros((3, 10, 10))
all_dis = []
for i in range(len(centers)):
    for j in range(len(centers[i])):
        for v in centers[i][j]:           
            min_dis = 10000
            for k in range(len(centers[i])):
                if k == j:
                    continue
                for vd in centers[i][k]:
                    dis = np.linalg.norm(v - vd)
                    if dis < min_dis:
                        min_dis = dis
            if min_dis > 20:
                min_dis = 20
            all_dis.append(min_dis)
numbers, _, _ = plt.hist(all_dis, bins=20, edgecolor='black', alpha=0.7, color = 'orange')
# print(np.sum(numbers[:10]) / len(all_dis))
plt.show()

np.savetxt("analysis1/output.txt", all_dis, fmt="%.4f", delimiter="\n")
