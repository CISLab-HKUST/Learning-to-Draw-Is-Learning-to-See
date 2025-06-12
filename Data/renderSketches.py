import json
import svgwrite
import numpy as np
import cairosvg
import os

trace_json = {}
with open('Data/eyetrack_drawing_with_fixation_point.json') as file:
    trace_json = json.load(file)

for uid, drawings in trace_json.items():
    for image, fixation_points in drawings.items():
        svg_filename = 'Data/sketches/' + image[:-4] + '_' + uid + '.svg'
        png_filename = 'Data/sketches_png/' + image[:-4] + '_' + uid + '.png'
        dwg = svgwrite.Drawing(filename=svg_filename, size=(800, 800))  
        for fixation_point in fixation_points:
            for stroke in fixation_point["strokes"]:
                txy = stroke["path"].split(",")
                width = stroke["width"]
                opacity = stroke["opacity"]
                color = stroke["color"]
                pressure_values = [float(p) for p in stroke["pressure"].split(",")]

                pressure_mean = np.mean(pressure_values)
                width = pressure_mean * width * 2

                if len(txy) <= 3:
                    continue     
                d = "M" + " ".join(f"{txy[3*vid+1]},{txy[3*vid+2]}" for vid in range(len(txy) // 3))
                w = dwg.path(d=d, fill="none", stroke=color, style=f"-webkit-tap-highlight-color: rgba(0, 0, 0, 0); stroke-linecap: round; stroke-linejoin: round; stroke-opacity: {opacity}; stroke-width: 1;")
                dwg.add(w)
        dwg.save()
        cairosvg.svg2png(url=svg_filename, write_to=png_filename, background_color='white')