import json
import numpy as np
import SimpleITK as sitk
from sklearn.cross_decomposition import CCA
import svgwrite
import cairosvg
import os

def read_outTx(data_name):
    transform_file_path = "Data/transform_labeled/transform_" # if isLabeled else "../transform_unlabeled/transform_"
    transform_file_path = transform_file_path + data_name[:-4] + ".txt"
    return sitk.ReadTransform(transform_file_path)

def point_transfer(src_point, outTx):
    downsample_src_point = src_point * 0.32
    downsample_dst_point = outTx.TransformPoint(downsample_src_point)
    return np.array(downsample_dst_point) * 3.125

def weight_sum(a, wa, b, wb):
    sum = a * wa + b * wb
    weight = wa + wb
    return sum, weight

# main
with open("Data/eyetrack_drawing_with_fixation_point.json", 'r') as eyetrack_data_file:
    eyetrack_data = json.load(eyetrack_data_file)
    registed_eyetrack_data = eyetrack_data
    
cr_count = 0
totoal_cr = 0
files = os.listdir('Data/sketches_png/') 
file_names = [file for file in files if file.endswith('png')]
for file_name in file_names:
    image = file_name[:-15] + '.png'
    uid = file_name[-14:-4]
    
    svg_filename = 'Data/registed_sketch/' + file_name[:-4] + '.svg'
    png_filename = 'Data/registed_sketch_png/' + file_name[:-4] + '.png'
    dwg = svgwrite.Drawing(filename=svg_filename, size=(800, 800))  
    fixation_datas = eyetrack_data[uid][image]
    outTx = read_outTx(file_name)
        # print(outTx.SetInverse())
    time = 0
    for fixation_data_index in range(len(fixation_datas)):
        fixation_data = fixation_datas[fixation_data_index]  
        strokes = fixation_data["strokes"]

        for stroke_index in range(len(strokes)):
            stroke = strokes[stroke_index]
            width = stroke["width"]
            opacity = stroke["opacity"]
            color = stroke["color"]
            pressure_values = [float(p) for p in stroke["pressure"].split(",")]

            pressure_mean = np.mean(pressure_values)
            width = pressure_mean * width * 2    
            d = "M"
            txy = stroke["path"].split(",")
            vertex_number = len(txy) // 3
                    
            for i in range(vertex_number):
                vertex = np.array([float(txy[3 * i + 1]), float(txy[3 * i + 2])])
                vertex = point_transfer(vertex, outTx)
                txy[3 * i + 1], txy[3 * i + 2] = str(vertex[0]), str(vertex[1])
                d = d + str(vertex[0]) + "," + str(vertex[1]) + " "
            w = dwg.path(d=d, fill="none", stroke=color, style=f"-webkit-tap-highlight-color: rgba(0, 0, 0, 0); stroke-linecap: round; stroke-linejoin: round; stroke-opacity: {opacity}; stroke-width: 2;")
            dwg.add(w)
            new_txy = ','.join(txy)
            registed_eyetrack_data[uid][image][fixation_data_index]["strokes"][stroke_index]["path"] = new_txy
                
    dwg.save()
    cairosvg.svg2png(url=svg_filename, write_to=png_filename, background_color='white')
        
with open("Data/registed_eyetrack_drawing_with_fixation_point.json", 'w') as registed_file:
    json.dump(registed_eyetrack_data, registed_file)
        #print("Canonical variables for X:\n", A)
        #print("Canonical variables for Y:\n", B)