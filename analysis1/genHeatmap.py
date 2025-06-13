import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import cv2


important_images = ["DIY_Gantry_bust_010_RGB.png", "PMS_Princeton_penguin1_000_RGBND.png", "WEB_CUHK_cottage1_000_RGB.png"]
with open('./Data/registed_eyetrack_drawing_with_fixation_point.json', 'r') as json_file:
    eyetrack_and_drawing_data = json.load(json_file)
files = os.listdir('Data/sketches_png/') 
file_names = [file for file in files if file.endswith('png')]
sps = [[],[],[]]
nsps = [[],[],[]]
p_names = [[],[],[]]
for uid, value in eyetrack_and_drawing_data.items():
    if uid == "eyetrack09":
        continue
    for image, fixation_datas in value.items():
        if image not in important_images:
            continue
        name = image[:-4] + '_' + uid + '.png'
        type = important_images.index(image)
        sp = np.zeros((16, 16))
        for fixation in fixation_datas:
            center = np.array([fixation["x"], fixation["y"]])
            tixel = np.floor(center / 50).astype(int)
            sp[tixel[1]][tixel[0]] += 1
        sps[type].append(sp)
        sp = sp / sp.sum()
        nsps[type].append(sp)
        p_names[type].append(name)
        fig = plt.figure(figsize=(10,10))
        plt.imshow(sp, cmap=plt.cm.coolwarm, vmin=0, vmax=0.10)
        
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        plt.axis('off')
        # plt.colorbar()
        plt.savefig('analysis1/heatmaps/' + name)  
        plt.close(fig)
        heat = cv2.imread('analysis1/heatmaps/' + name, cv2.IMREAD_COLOR)
        origin = cv2.imread('Data/select_image/' + image, cv2.IMREAD_COLOR)
        heat = cv2.resize(heat, (origin.shape[1], origin.shape[0]))
        alpha = 0.6
        combined = cv2.addWeighted(origin, 1 - alpha, heat, alpha, 0)
        cv2.imwrite('analysis1/heatmaps/' + name[:-4] + '_combined.png', combined)
for i in range(3):
    spt = np.sum(sps[i], axis=0)
    spt = spt / spt.sum()
    # np.save('userStudy/' + important_images[i][:-4] + '.npy', spt)
    fig = plt.figure(figsize=(10,10))
    plt.imshow(spt, cmap=plt.cm.coolwarm, vmin=0, vmax=0.10)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')
    plt.savefig('analysis1/heatmaps/' + important_images[i])  
    plt.close(fig)
    heat = cv2.imread('analysis1/heatmaps/' + important_images[i], cv2.IMREAD_COLOR)
    origin = cv2.imread('Data/select_image/' + important_images[i], cv2.IMREAD_COLOR)
    heat = cv2.resize(heat, (origin.shape[1], origin.shape[0]))
    alpha = 0.6
    combined = cv2.addWeighted(origin, 1 - alpha, heat, alpha, 0)
    cv2.imwrite('analysis1/heatmaps/' + important_images[i][:-4] + '_combined.png', combined)
    for j in range(len(nsps[i])):
        temp_spt = []
        temp_nsps = []
        f1 = spt.flatten()
        f2 = nsps[i][j].flatten()
        for k in range(len(f1)):
            if f1[k] != 0 or f2[k] != 0:
                temp_spt.append(f1[k])
                temp_nsps.append(f2[k])
        correlation, p_values = pearsonr(spt.flatten(), nsps[i][j].flatten())
        l2_loss = np.sqrt(np.sum((spt - nsps[i][j]) ** 2))
        name = p_names[i][j]
        print(name + ': ' + str(correlation))
    fig = plt.figure(figsize=(10,10))
    # average
    plt.imshow(spt, cmap=plt.cm.coolwarm, vmin=0, vmax=0.10)
    plt.axis('off')
    plt.colorbar()
    plt.savefig('analysis1/heatmaps/c_' + important_images[i])  
    plt.close(fig)

