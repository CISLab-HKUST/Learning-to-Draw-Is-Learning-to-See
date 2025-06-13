import numpy as np
import cv2
import os
from scipy.spatial import KDTree

OUT_GROUP_TIMES_THRESHOLD = 40
CHECKED_RATE_THRESHOLD = 0.5
VISION_RADIUS_THRESHOLD = 60

def getUid(image):
    data_file = 'Data/image-for-final-project/'
    sketch_svgs = [sketch_svg for sketch_svg in os.listdir(data_file) if sketch_svg.endswith(".svg") and sketch_svg.startswith(image[:-4])]
    assert(len(sketch_svgs) > 0)
    sketch_svg = sketch_svgs[0]
    uid = sketch_svg[-14:-4]
    return uid

class Checkpoint:
    def __init__(self, time, groupId):
        self.time = time
        self.groupId = groupId
        self.isChecked = False
        self.radius = 0
        
    def setRadius(self, r):
        self.radius = r
        
class Group:
    def __init__(self, startId, endId):
        self.startId = startId
        self.endId = endId
        self.point_count = endId - startId
        self.checked_count = 0
        self.startTime = 0
        self.endTime = 0
        self.localKDTree = None
        self.aabb = AABB(0, 0)
        self.aabb.initAsEmpty()
    
    def getCheckedRate(self):
        return self.checked_count / self.point_count
    
    def getAverageTime(self):
        return (self.startTime + self.endTime) / 2

class SketchCheckpoints:
    def __init__(self, json_points):
        points = []
        points_pos = []
        groups = []
        cur_groupId = -1
        startId = -1
        for json_point_index in range(len(json_points)):
            json_point = json_points[json_point_index]
            point_pos = np.array([json_point["x"], json_point["y"]])
            groupId = json_point["groupId"]
            point = Checkpoint(json_point["time"], groupId)
            points.append(point)
            points_pos.append(point_pos)
            # check new group
            if cur_groupId == -1:
                cur_groupId = groupId
                startId = 0
            elif groupId != cur_groupId:
                groups.append(Group(startId, json_point_index))
                cur_groupId = groupId
                startId = json_point_index
        groups.append(Group(startId, len(json_points)))        
        self.checkpoints = points
        self.checkpoints_pos = points_pos
        
        for group in groups:
            group.startTime = points[group.startId].time
            group.endTime = points[group.endId].time if group.endId < len(json_points) else points[group.endId - 1].time
            group.localKDTree = KDTree(self.checkpoints_pos[group.startId : group.endId])
            for pos_index in range(group.startId, group.endId):
                group.aabb.combine(AABB(points_pos[pos_index][0], points_pos[pos_index][1]))
                
        self.checkpoints = points
        self.checkpoints_pos = points_pos
        self.pointGroups = groups
        self.kd_tree = KDTree(self.checkpoints_pos)
        self.checkRadius = 20 # default
        self.currentGroupId = 0
        self.outGroupTimes = 0
        print(len(self.pointGroups))
        print(self.checkpoints[-1].groupId)
        assert(len(self.pointGroups) == self.checkpoints[-1].groupId + 1)
    
    def findNextGroup(self, curGId, testPoint):
        # make sure current group has been finished
        if self.pointGroups[curGId].getCheckedRate() <= CHECKED_RATE_THRESHOLD:
            return curGId
        candidateGId = [None, None]
        if curGId > 0:
            for gid in range(curGId - 1, -1, -1):
                if self.pointGroups[gid].getCheckedRate() <= CHECKED_RATE_THRESHOLD:
                    candidateGId[0] = gid
                    break
        if curGId < len(self.pointGroups) - 1:
            for gid in range(curGId + 1, len(self.pointGroups)):
                if self.pointGroups[gid].getCheckedRate() <= CHECKED_RATE_THRESHOLD:
                    candidateGId[1] = gid
                    break
        if (candidateGId[0] is None) and (candidateGId[1] is None):
            return -1
        elif candidateGId[0] is None:
            return candidateGId[1]
        elif candidateGId[1] is None:
            return candidateGId[0]
        else:
            leftDistance, _pad = self.pointGroups[candidateGId[0]].localKDTree.query(testPoint)
            rightDistance, _pad = self.pointGroups[candidateGId[1]].localKDTree.query(testPoint)
            return candidateGId[0] if leftDistance < rightDistance else candidateGId[1]
    
    def moveToNextGroup(self, nextGId):
        assert(self.currentGroupId != nextGId)
        assert(nextGId != -1)
        self.currentGroupId = nextGId
        self.outGroupTimes = 0
    
    def getCurrentArea(self):
        center, radius = self.pointGroups[self.currentGroupId].aabb.getCircle()
        radius = radius if radius > VISION_RADIUS_THRESHOLD else VISION_RADIUS_THRESHOLD
        return center, radius
     
    def testCheckpoint(self, query_point):
        currentGId = self.currentGroupId
        distance, index = self.kd_tree.query(query_point)
        # update checkpoint and group
        activate_groupId = self.checkpoints[index].groupId
        if distance < self.checkRadius:
            has_checked = self.checkpoints[index].isChecked
            if not has_checked:
                self.pointGroups[activate_groupId].checked_count += 1
                self.checkpoints[index].isChecked = True
        # update currentGroup
        if activate_groupId != currentGId:
            print("ac: "+ str(activate_groupId) +"; cu: " + str(currentGId))
            # move to activate group
            if self.outGroupTimes >= OUT_GROUP_TIMES_THRESHOLD: 
                currentGId = activate_groupId
            # check if it is neighbor point
            else:
                min_distance, _pad = self.pointGroups[currentGId].localKDTree.query(query_point)
                if min_distance < 2 * self.currentGroupId: # neighbor point: stay in current group
                    self.outGroupTimes += 1
                    currentGId = activate_groupId
                else: # np.abs(activate_groupId - currentGId) <= 2: # not neighbor point: move to activate group
                    currentGId = activate_groupId
        else: print("cu: " + str(currentGId))
        nextGId = self.findNextGroup(currentGId, query_point)
        if nextGId < 0:
            return None, None
        elif nextGId != self.currentGroupId:
            self.moveToNextGroup(nextGId)
        return self.getCurrentArea()
        
    
        

class Combined_fixation_data:
    def __init__(self):
        self.time = 0
        self.center = np.array([0, 0])
        self.strokes = []
        self.startTime = 10**20
        self.endTime = 0
    
    def add(self, temp_center, start, end, new_strokes):
        temp_time = end - start
        center, self.time = weight_sum(self.center, self.time, temp_center, temp_time)
        self.center = center / self.time
        self.strokes = self.strokes + new_strokes
        self.startTime = min(self.startTime, start)
        self.endTime = max(self.endTime, end)
        return len(self.strokes)

    def getPointCenter(self):
        count = 0
        x = 0
        y = 0
        for stroke in self.strokes:
            txy = stroke["path"].split(",")
            vertex_number = len(txy) // 3
            count = count + vertex_number
            for i in range(vertex_number):
                x = x + float(txy[3 * i + 1])
                y = y + float(txy[3 * i + 2])
        return np.array([x / count, y / count])        
    
    def getStrokesTime(self):
        start = 10**14
        end = 0
        for stroke in self.strokes:
            txy = stroke["path"].split(",")           
            vertex_number = len(txy) // 3
            start = min(int(txy[0]), start)
            end = max(int(txy[3 * (vertex_number - 1)]), end)
        return np.array([start, end]) 
        
        
class AABB:    
    def __init__(self, x, y):
        vertex = np.array([x, y])
        self.topLeft = vertex
        self.downRight = vertex
    
    def initAsEmpty(self):
        self.topLeft = np.array([float('inf'), float('inf')])
        self.downRight = np.array([float('-inf'), float('-inf')])
        
    def combine(self, aabb):
        self.topLeft = np.minimum(self.topLeft, aabb.topLeft)
        self.downRight = np.maximum(self.downRight, aabb.downRight)

    def getCircle(self):
        center = (self.topLeft + self.downRight) / 2
        radius = np.linalg.norm(self.topLeft - center)
        return center, radius
    

def txy2AABB(txy):
    vertex_number = len(txy) // 3
    final_aabb = AABB(0, 0)
    final_aabb.initAsEmpty()
    for i in range(vertex_number):
        x = float(txy[3 * i + 1])
        y = float(txy[3 * i + 2])
        aabb = AABB(x, y)
        final_aabb.combine(aabb)
    return final_aabb

def render_heatmap(heat_map, image):
    
    max_data = np.max(heat_map)
    heat_map = heat_map / max_data

    # 将数据标准化到0-255之间
    data_normalized = cv2.normalize(heat_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 使用伪彩色映射
    color_map = cv2.applyColorMap(data_normalized, cv2.COLORMAP_JET)  # 使用 JET 颜色映射

    image_path = 'C:/Users/14113/Desktop/phd/sketch/final project/eyetrack/src/assets/images/' + image  # 替换为您的图片路径
    #print(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # 将目标图片转换为浮点型，以便进行加权
    image_float = image.astype(np.float32)
    #print(image_float.size)

    # 将热力图转换为 float32 类型以确保类型一致
    heatmap_float = color_map.astype(np.float32)
    #print(heatmap_float.size)
    # 设置 alpha 值
    alpha = 0.4
    beta = 1 - alpha  # beta 值

    # 叠加热力图和目标图片
    combined = cv2.addWeighted(heatmap_float, alpha, image_float, beta, 0)
    combined_result = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return combined_result        

def render_fixation_point(center, time):
    local_heat_map = np.zeros((800, 800))
    sigma = 15
    three_sigma = 3 * sigma
    bounding_box_top_left = np.round(np.maximum(center - three_sigma, 0))
    bounding_box_down_right = np.round(np.minimum(center + three_sigma, 799))
    for i in range(int(bounding_box_top_left[0]), int(bounding_box_down_right[0])):
        for j in range(int(bounding_box_top_left[1]), int(bounding_box_down_right[1])):
            pixel_point = np.array([i, j]) + 0.5
            distance = np.linalg.norm(pixel_point - center)
            if distance <= three_sigma:
                local_heat_map[j][i] += time * np.exp(-(distance * distance) / (2 * sigma * sigma)) / (2 * np.pi * sigma * sigma)
    return local_heat_map

def weight_sum(a, wa, b, wb):
    sum = a * wa + b * wb
    weight = wa + wb
    return sum, weight

def combine_fixation_points(fixation_datas):
    combined_fixation_datas = []
    fd = Combined_fixation_data()
    for fixation_data in fixation_datas:
        temp_center = np.array([fixation_data["x"], fixation_data["y"]])
        strokes = fixation_data["strokes"]
        stroke_num = fd.add(temp_center, fixation_data["startTime"], fixation_data["endTime"], strokes)
        if stroke_num != 0: 
            combined_fixation_datas.append(fd)
            fd = Combined_fixation_data()
    return combined_fixation_datas

