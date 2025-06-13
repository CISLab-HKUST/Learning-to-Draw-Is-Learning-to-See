<<<<<<< HEAD
# Learning to Draw Is Learning to See: Analyzing Eye Tracking Patterns for Assisted Observational Drawing
[Paper](https://cislab.hkust-gz.edu.cn/media/documents/_SIGGRAPH_2025__Learning_to_Draw_Is_Learning_to_See_4.pdf) | [Project]() | [Data](https://github.com/CISLab-HKUST/Learning-to-Draw-Is-Learning-to-See) | [Interface](https://github.com/CISLab-HKUST/Learning-to-Draw-Is-Learning-to-See_Interface)

[中文](README_CN.md)

This repository contains the data and analysis programs for our paper "Learning to Draw Is Learning to See: Analyzing Eye Tracking Patterns for Assisted Observational Drawing" published at SIGGRAPH 2025.

## Getting Started
Create a virtual environment:
```
python -m venv venv
```
Activate the virtual environment:
```
.\venv\Scripts\activate
```
Install dependencies:
```
pip install -r .\requirements.txt
```

## Data Structure
We collected eye tracking data using the Tobii Pro Spark eye tracker and synchronized stroke data from users using Tracer. The two datasets were paired based on timestamps to form Fixation-strokes pairs. The Fixation-strokes pairs data are saved in the `eyetrack_drawing_with_fixation_point.json`file, with the following format:
```
{
    // each artist id
    "eyetrack00": {
        // each image prompt
        "DIY_Gantry_bust_010_RGB.png": [
            // fixation points list
            {
                "startTime": unit // The timestamp when fixation point start.
                "endTime": uint // The timestamp when fixation point end.
                "timeSpan": uint // endTime - startTime
                "x": float // The x coordinate of the fixation point.
                "y": float // The y coordinate of the fixation point.
                "strokes": [
                    // The subsequent strokes drawn from the start time of this fixation point
                    // to the start time of the next fixation point.
                    {
                        "path": string (Unix timestamp, x, y coordinates at each vertex separated by comma)
                        "pressure": string (pressure value at each vertex separated by comma)
                        "color": string (hex code, e.g., "#000000")
                        "width": integer (stroke width on an 800x800 canvas)
                        "opacity": float (alpha value from 0 to 1)
                    }
                    ...
                ]
            }
            ...
        ]
    }
}
```

## Stroke Registration
Our stroke registration section mainly references[Trace-vs-Freehand](https://github.com/zachzeyuwang/tracing-vs-freehand). Unlike that project, we do not have trace data as the target. Instead, we use the sketches extracted from image prompts by [informative-drawings](https://github.com/carolineec/informative-drawings) as the target.

Create the following folders under the `Data` folder:
```
- Data/
  - anime_style/
  - sketches/
  - sketches_png/
  - transform_labeled/
  - visual_labeled/
  - select_image/
  - ...
```
Download all image prompts from [here](https://drive.google.com/file/d/1G68q0oulKKFKlLSGv_D1ONNgfn74hAgs/view?usp=sharing) and place them in the `select_image/`.

Use the [informative-drawings](https://github.com/carolineec/informative-drawings) repository to extract `anime_style` sketches from the image prompts and place them in the `anime_style/`folder. Alternatively, you can download the pre-extracted results from [here](https://drive.google.com/file/d/1RCIRJnGBravQDkw0OHgwNWBtM83jRRKG/view?usp=sharing).

Render the sketches:
```
python Data/renderSketches.py
```
Render all sketches drawn by artists recorded in `eyetrack_drawing_with_fixation_point.json` as SVG and PNG files, and save them in the `sketches/` and `sketches_png/` folders, respectively.

Stroke registration:
```
.\register_labeled.bat
```

Generate registered Fixation-strokes pairs:
```
python Data/generateRegisteredPairs.py
```

Evaluate the accuracy of artists' sketches (Fig. 8 AVGD):
```
python Data/accurancy.py
```

## Do People Focus on Similar Areas?
Generate fixation point heatmaps and correlation coefficients (Fig. 2):
```
python .\analysis1\genHeatmap.py
```
Heatmap results are saved in the `analysis1/heatmaps/`, and correlation coefficients are output to the console.

Calculate the minimum distance histogram of fixation points (Fig. 3):
```
python .\analysis1\minDistance.py
```

## Do People Draw Where They Observe?
Canonical Correlation Analysis：
```
python .\analysis2\cca.py
```
Scatter plots are saved in `analysis2/point-time/`, and correlation coefficients and $p$-values are output to the console. Fig. 4 is the combined result of `analysis2/point-time/blue_DIY_Gantry_bust_010_RGB_eyetrack09.png` and `analysis2/point-time/red_DIY_Gantry_bust_010_RGB_eyetrack09.png`.

Multivariate Linear Mixed-Effects Model：
```
python .\analysis2\MLMM.py
```
Results are output to the console.

## How Do People Observe and Draw Over Time
Canonical Correlation Analysis in first window：
```
python .\analysis3\point-time-cca.py
```
Scatter plots are saved in `analysis3/stroke-point-time/`, and correlation coefficients and $p$-values are output to the console.

Slide window:
```
python .\analysis3\window.py
```
Scatter plots are saved in `analysis3/window/`. Fig. 6a is `analysis3\window\WEB_CUHK_man_000_RGB_eyetrack05.png`.



=======
# Learning-to-Draw-Is-Learning-to-See
Coming soon
>>>>>>> f0d98f16884bd1c30d633c841076bcde5e1345b6
