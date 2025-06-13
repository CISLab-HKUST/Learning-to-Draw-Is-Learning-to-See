import os
import json
import fixation_data_utils as f_utils
import numpy as np
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
import cv2
from scipy.stats import pearsonr
from scipy.spatial import ConvexHull

window_size = 0.2

def getWindowEndIndex(times, windowStartTime, startIndex):
    for i in range(startIndex, len(times), 1):
        time = times[i]
        if time > windowStartTime + window_size:
            return i
    return len(times)

def getWindowIndexRange(times, windowStartTime):
    state = 0
    startIndex = 0
    endIndex = len(times)
    for i in range(len(times)):
        if state == 0 and times[i] > windowStartTime:
            state = 1
            startIndex = i
        if state == 1 and times[i] > windowStartTime + window_size:
            return startIndex, i
    return startIndex, endIndex

data_file = 'Data/sketches/'
sketch_svgs = [sketch_svg for sketch_svg in os.listdir(data_file) if sketch_svg.endswith(".svg")]
with open('./Data/registed_eyetrack_drawing_with_fixation_point.json', 'r') as json_file:
    eyetrack_and_drawing_data = json.load(json_file)

avg_area = 0 
a_count = 0
avg_t = 0
a = []
outline_t = []
for sketch_svg in sketch_svgs:
    image = sketch_svg[:-15] + '.png'
    
    # for fig-3b
    # if image != "WEB_CUHK_man_000_RGB.png":
    #     continue
    
    uid = sketch_svg[-14:-4]
    points = []
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
    # original_strokes_centers = strokes_centers.copy()
    strokes_time = [(time[0] + time[1]) / 2 for time in strokes_time]
    project_pos, B = cca.transform(strokes_centers, strokes_time)
    project_pos = project_pos.T if correlation > 0 else -project_pos.T
    # print(project_pos)
    cc = []
    t = []
    p = []
    for windowStartTime in np.arange(0, 0.8, 0.01):
        startIndex, endIndex = getWindowIndexRange(strokes_time, windowStartTime)
        if (endIndex - startIndex < 2):
            cc.append(cc[len(cc) - 1])
            t.append(windowStartTime)
            continue
        window_centers = project_pos[0][startIndex:endIndex]
        window_times = strokes_time[startIndex:endIndex]
        correlation, p_values = pearsonr(window_centers, window_times)
        cc.append(correlation)
        t.append(windowStartTime)
        p.append(p_values)
        if endIndex == len(strokes_centers):
            break
    '''   
    for index in range(len(strokes_centers)):
        windowStartTime = strokes_time[index]
        windowEndIndex = getWindowEndIndex(strokes_time, windowStartTime, index)
        if (windowEndIndex - index < 2):
            continue
        window_centers = project_pos[0][index:windowEndIndex]
        window_times = strokes_time[index:windowEndIndex]
        correlation, p_values = pearsonr(window_centers, window_times)
        cc.append(correlation)
        t.append(windowStartTime)
        p.append(p_values)
        if windowEndIndex == len(strokes_centers):
            break
    ''' 
    # plt.title(sketch_svg[:-4])
    #np.savetxt("analysis3/cc_t.txt", t, delimiter="\n")
    #np.savetxt("analysis3/cc.txt", cc, delimiter="\n")
    plt.plot(t,cc,marker='o', label = 'correlation coefficients')
    plt.xlabel('time')
    plt.ylabel('c')
    plt.grid()
    areas = []
    a_t = []
    for combined_fixation_data_index in range(len(combined_fixation_datas)):
        combined_fixation_data = combined_fixation_datas[combined_fixation_data_index]
        for stroke in combined_fixation_data.strokes:
            txy = stroke["path"].split(",")
            for vi in range(len(txy) // 3):
                points.append(np.array([float(txy[3 * vi + 1]), float(txy[3 * vi + 2])]))
        if len(points) < 3:
            areas.append(0)
            a_t.append(strokes_time[combined_fixation_data_index])
            continue
        hull = ConvexHull(points)
        points = [points[v] for v in hull.vertices]
        areas.append(hull.area)
        a_t.append(strokes_time[combined_fixation_data_index])
    areas = np.array(areas) / np.max(areas)
    a_t = np.array(a_t)
    #np.savetxt("analysis3/a_t.txt", a_t[a_t<=0.8], delimiter="\n")
    #np.savetxt("analysis3/a.txt", areas[:len(a_t[a_t<0.8])], delimiter="\n")
    plt.plot(a_t,areas[:len(a_t)],marker='x', label = 'convex hull area ratio')
    plt.legend()
    plt.savefig('analysis3/window/' + sketch_svg[:-4] + '.png')
    plt.close()
    state = 0
    finish = -1
    for i in range(len(cc)):
        if state == 0:
            if cc[i] < -0.5:
                state = 1
        if state == 1:
            if cc[i] > 0.25:
                finish = i
                break
    if finish > 0 and finish < len(t):
        for i in range(len(a_t)):
            if a_t[i] > t[finish]:
                avg_area += areas[i]
                a.append(areas[i])
                print(sketch_svg[:-4] + ':' + str(areas[i]) + ',  ' + str(i))
                avg_t += a_t[i]
                outline_t.append(a_t[i])
                a_count += 1
                break
        
    else:
        print(sketch_svg[:-4] + ':')
num,_,_=plt.hist(a, bins=[0,0.8,0.9,1], edgecolor='black', alpha=0.7, color = 'green')
a = np.array(a)
#np.savetxt("ratio.txt", a, delimiter="\n")
#np.savetxt("time.txt", outline_t, delimiter="\n")

print(np.mean(a))
print(np.var(a))
print(avg_t / a_count)    
plt.close()
plt.scatter(outline_t, a)
plt.ylabel('convex hull area ratio')
plt.xlabel('time')
plt.grid()
plt.savefig('analysis3/outline.svg')