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

important_images = ["DIY_Gantry_bust_010_RGB.png", "PMS_Princeton_penguin1_000_RGBND.png", "WEB_CUHK_cottage1_000_RGB.png"]
ac_json = {}
def objective(x, v, v_prime):
    s, d_x, d_y = x
    d = np.array([d_x, d_y])
    return np.sum(np.linalg.norm(s * v + d - v_prime, axis=1))

with open('./Data/registed_eyetrack_drawing_with_fixation_point.json', 'r') as r_json_file:
    registed_eyetrack_and_drawing_data = json.load(r_json_file)
with open('./Data/eyetrack_drawing_with_fixation_point.json', 'r') as json_file:
    eyetrack_and_drawing_data = json.load(json_file)

def getSketchCenter(fixation_datas):
    barycenter = np.array([0, 0])
    count = 0
    for fixation_data in fixation_datas:
        for stroke in fixation_data.strokes:
            txy = stroke["path"].split(",")
            for vi in range(len(txy) // 3):
                barycenter[0] += float(txy[3 * vi + 1])
                barycenter[1] += float(txy[3 * vi + 2])
                count += 1
    if count == 0:
        return np.array([0, 0])
    barycenter = barycenter / count
    return barycenter

for uid, value in eyetrack_and_drawing_data.items():
    avg_dis = 0
    count = 0
    for image, fixation_datas in value.items():
        if image not in important_images:
           continue
        registed_fixation_datas = registed_eyetrack_and_drawing_data[uid][image]
        assert(len(registed_fixation_datas) == len(fixation_datas))
        combined_fixation_datas = f_utils.combine_fixation_points(fixation_datas)
        registed_combined_fixation_datas = f_utils.combine_fixation_points(registed_fixation_datas)
        name = image[:-4] + '_' + uid + '.png'
        barycenter = getSketchCenter(combined_fixation_datas)
        registed_barycenter = getSketchCenter(registed_combined_fixation_datas)
        offset = barycenter - registed_barycenter
        vertices = []
        r_vertices = []
        for combined_fixation_data_index in range(min(len(combined_fixation_datas), 10000)):
            combined_fixation_data = combined_fixation_datas[combined_fixation_data_index]
            r_combined_fixation_data = registed_combined_fixation_datas[combined_fixation_data_index]
            strokes = combined_fixation_data.strokes
            r_strokes = r_combined_fixation_data.strokes

            
            for stroke in strokes:
                txy = stroke["path"].split(",")
                for vi in range(len(txy) // 3):
                    vertices.append(np.array([float(txy[3 * vi + 1]), float(txy[3 * vi + 2])]))
            for r_stroke in r_strokes:
                r_txy = r_stroke["path"].split(",")
                for vi in range(len(r_txy) // 3):
                    r_vertices.append(np.array([float(r_txy[3 * vi + 1]), float(r_txy[3 * vi + 2])]))
            assert(len(vertices) == len(r_vertices))    
        vertices = np.array(vertices)
        r_vertices = np.array(r_vertices)
        if len(vertices) < 1: 
            continue
        x = np.array([1, offset[0], offset[1]])
        # print(name + " before:" + str(objective(x, vertices, r_vertices) / len(vertices)))
        res = minimize(objective, x, args=(vertices, r_vertices), method='SLSQP', constraints=())
        dis = objective(res.x, vertices, r_vertices) / len(vertices)
        print(name + ': ' + str(dis))
        
        