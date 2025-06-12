## 开始
创建虚拟环境:
```
python -m venv venv
```
激活虚拟环境:
```
.\venv\Scripts\activate
```
安装依赖:
```
pip install -r .\requirements.txt
```

## 数据结构
我们使用Tobii Pro Spark眼动追踪仪收集眼动数据，使用Tracer同步收集用户的笔画数据，并根据时间戳将二者构建为Fixation-strokes pairs. 
Fixation-strokes pairs数据保存于`eyetrack_drawing_with_fixation_point.json`文件中，格式如下：
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

## 笔画注册
我们的笔画注册部分主要参考了[Trace-vs-Freehand](https://github.com/zachzeyuwang/tracing-vs-freehand). 与之不同的是，我们没有Trace数据作为目标，而是使用[informative-drawings](https://github.com/carolineec/informative-drawings)从image prompt中提取的草图作为目标。

在`Data`文件夹下新建`anime_style`,`sketches`,`sketches_png`,`transform_labeled`,`visual_labeled`,`select_image`文件夹：
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
从[这里](https://drive.google.com/file/d/1G68q0oulKKFKlLSGv_D1ONNgfn74hAgs/view?usp=sharing)下载所有的image prompts并放入`select_image/`文件夹下。

使用[informative-drawings](https://github.com/carolineec/informative-drawings)从image prompt中提取的`anime_style`风格的草图放入`anime_style/`文件夹下。或者从[这里](https://drive.google.com/file/d/1RCIRJnGBravQDkw0OHgwNWBtM83jRRKG/view?usp=sharing)下载我们已经提取的结果。

渲染草图：
```
python Data/renderSketches.py
```
将`eyetrack_drawing_with_fixation_point.json`中记录的所有艺术家绘制的草图渲染为svg和png文件，分别保存到`sketches/`和`sketches_png/`文件夹下

笔画注册：
```
.\register_labeled.bat
```

生成注册后的Fixation-strokes pairs：
```
python Data/generateRegisteredPairs.py
```

## Do People Focus on Similar Areas?