import os
import json
import fixation_data_utils as f_utils
import numpy as np
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
import cv2
from scipy.stats import pearsonr

data_file = 'Data/sketches/'
sketch_svgs = [sketch_svg for sketch_svg in os.listdir(data_file) if sketch_svg.endswith(".svg")]
with open('./Data/registed_eyetrack_drawing_with_fixation_point.json', 'r') as json_file:
    eyetrack_and_drawing_data = json.load(json_file)
    
p_count = 0
c_count = [0, 0, 0, 0]
p = []
cc = []
both_count = 0
for sketch_svg in sketch_svgs:
    image = sketch_svg[:-15] + '.png'
    uid = sketch_svg[-14:-4]
    
    fixation_datas = eyetrack_and_drawing_data[uid][image]
    combined_fixation_datas = f_utils.combine_fixation_points(fixation_datas)

    strokes_centers = [fixation_data.getPointCenter() for fixation_data in combined_fixation_datas]
    if (len(strokes_centers) == 0):
        continue
    # original_strokes_centers = strokes_centers.copy()
    strokes_time = [fixation_data.getStrokesTime() for fixation_data in combined_fixation_datas]
    strokes_time = (strokes_time - np.min(strokes_time)) / (np.max(strokes_time) - np.min(strokes_time))
    
    pre_strokes_time = [(time[0] + time[1]) / 2 for time in strokes_time if time[0] < 0.2]
    pre_strokes_centers = strokes_centers[:len(pre_strokes_time)]
    if (len(pre_strokes_centers) < 2):
        continue
    cca = CCA(n_components=1, max_iter=1000)
    cca.fit(pre_strokes_centers, pre_strokes_time)
    A, B = cca.transform(pre_strokes_centers, pre_strokes_time)
    correlation, p_values = pearsonr(A.T[0], pre_strokes_time)
    correlation = abs(correlation)
    cc.append(correlation)
    p.append(p_values)
    if correlation > 0.6 and p_values < 0.05:
        both_count += 1
        
    # 绘制散点图
    print((correlation, p_values))
    for x, y in zip(A, pre_strokes_time):
        # 绘制线段
        plt.plot(x, y, color='blue', marker='o')

    # 添加标签和标题
    vector_name = 'b'
    plt.title(sketch_svg[:-4])
    plt.xlabel('Projected positon.' + vector_name + ':' + str((cca.x_rotations_.tolist())))
    plt.ylabel('time')
    plt.grid()
    plt.savefig('analysis3/stroke-point-time/' + sketch_svg[:-4] + '.png')
    plt.close()
    
#np.savetxt("analysis3/cc.txt", cc, delimiter="\n")
#np.savetxt("analysis3/p.txt", p, delimiter="\n")

num_cc,_,_ = plt.hist(cc, bins=[0, 0.6, 0.8, 1], edgecolor='black', alpha=0.7, color = 'red')
# np.savetxt("analysis3/cc.txt", cc, delimiter="\n")
#plt.show()
#bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
num_p,_,_ =plt.hist(p, bins=[0,0.01,0.05,1], edgecolor='black', alpha=0.7, color = 'orange')
# np.savetxt("analysis3/p.txt", p, delimiter="\n")
#plt.show()
plt.close()
plt.scatter(cc, p)
plt.ylabel('p-value')
plt.xlabel('correlation coefficient')
plt.axvline(x=0.6, color='red', linestyle='--', linewidth=1, label='correlation coefficient is 0.6')
# 添加水平虚线
plt.axhline(y=0.05, color='green', linestyle='--', linewidth=1, label='p-value is 0.05')

plt.savefig('analysis3/p_c.svg')
print(both_count)
print(len(p))