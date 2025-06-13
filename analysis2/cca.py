import os
import json
import fixation_data_utils as f_utils
import numpy as np
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
import cv2
from scipy.stats import pearsonr

def plot_range_with_lines(A, B, name, color='blue', projection_vector=None):
    
    if len(A) != len(B):
        raise ValueError("Arrays A and B must be of the same length.")
    for x, y in zip(A, B):
        plt.plot((y[0] + y[1]) / 2, x, color=color, marker='o')

    vector_name = 'a' if color == 'red' else 'b'
    # plt.title(name)
    plt.ylabel('Projected positon.')# + vector_name + ':' + str((projection_vector.tolist())))
    # print(name + ' ' + vector_name + ':' + str((projection_vector.tolist())))
    plt.xlabel('time')
    plt.grid()
    plt.savefig('analysis2/point-time/' + color + '_' + name)
    plt.close()
    # np.savetxt("analysis2/" + color + "_pos_" + name + ".txt", A, delimiter="\n")
    # np.savetxt("analysis2/" + color + "_time_" + name + ".txt", Bn, delimiter="\n")
    

data_file = 'Data/sketches/'
sketch_svgs = [sketch_svg for sketch_svg in os.listdir(data_file) if sketch_svg.endswith(".svg")]
with open('./Data/registed_eyetrack_drawing_with_fixation_point.json', 'r') as json_file:
    eyetrack_and_drawing_data = json.load(json_file)
    
p_count = 0
c_count = [0, 0, 0, 0]
p = []

plt.rcParams['font.family'] = 'Times New Roman'  
for sketch_svg in sketch_svgs:
    image = sketch_svg[:-15] + '.png'
    uid = sketch_svg[-14:-4]
    
    # for paper
    # if sketch_svg != "DIY_Gantry_bust_010_RGB_eyetrack09.svg" and sketch_svg != "DIY_Gantry_teapot_020_RGB_eyetrack04.svg":
    #     continue
    
    fixation_datas = eyetrack_and_drawing_data[uid][image]
    combined_fixation_datas = f_utils.combine_fixation_points(fixation_datas)
    
    fixation_centers = [fixation_data.center for fixation_data in combined_fixation_datas]
    
    fixation_time = [np.array([fixation_data.startTime, fixation_data.endTime]) for fixation_data in combined_fixation_datas]
    if (len(fixation_time) == 0):
        continue
    fixation_time[len(fixation_time)-1][1] = fixation_time[len(fixation_time)-1][0]
    fixation_time[0][0] = fixation_time[0][1]
    # print(np.min(fixation_time))
    fixation_time = (fixation_time - np.min(fixation_time)) / (np.max(fixation_time) - np.min(fixation_time))

    strokes_centers = [fixation_data.getPointCenter() for fixation_data in combined_fixation_datas]
    original_strokes_centers = strokes_centers.copy()
    strokes_time = [fixation_data.getStrokesTime() for fixation_data in combined_fixation_datas]
    strokes_time = (strokes_time - np.min(strokes_time)) / (np.max(strokes_time) - np.min(strokes_time))
    
    max_time = np.max(strokes_time)
    #fixation_time = (fixation_time - np.min(fixation_time)) / (max_time - np.min(fixation_time))
    #strokes_time = (strokes_time - np.min(strokes_time)) / (max_time - np.min(strokes_time))
    # print(image)
    fixation_centers = np.array(fixation_centers)
    strokes_centers = np.array(strokes_centers)
    cca = CCA(n_components=1)
    cca.fit(fixation_centers, strokes_centers)
    A, B = cca.transform(fixation_centers, strokes_centers)
    correlation, p_values = pearsonr(A.T[0], B.T[0])
    print(sketch_svg[:-4] + ": ", (correlation, p_values))
    plot_range_with_lines(A, fixation_time, sketch_svg[:-4], 'red', cca.x_rotations_)
    plot_range_with_lines(B, strokes_time, sketch_svg[:-4], 'blue', cca.y_rotations_)
    # draw_vector(original_strokes_centers, image)
    
