# AD/PD三个指标测量

本仓库主要用于zEvans, Evans, BVR, CA四个指标测量，目前模型仍在进一步优化中......

## TODO
- [ ] 分割模型持续优化
- [ ] CA角度测量方法改进
- [ ] 识别测量Evans指数的横断面的模型有待进一步优化

## 环境搭建
本项目基于Python环境，运行下述命令安装所需python库
```
pip install -r requirements.txt
```
## 模型准备
模型文件地址：[Google Drive](https://drive.google.com/drive/folders/1KyrTW64qj_ZCg0NHJJafLGa6cvdNWoIv?usp=drive_link)

## 数据准备
将dicom格式的数据组织成如下形式：
```
├── dicom_dir
│   ├── dicom_1
│   │   ├── 00001.dcm
│   │   ├── 00002.dcm
│   │   ├── 00003.dcm
│   │   ├── 00004.dcm
│   │   ├── 00005.dcm
│   │   ├── 00011.dcm
│   │   └── ...
│   └── dicom_2
│       ├── IM000000
│       ├── IM000001
│       ├── IM000002
│       ├── IM000003
│       ├── IM000004
│       ├── IM000005
│       ├── ...

```
## 运行指令
```
python main.py \
--dicom_dir # 输入的dicom文件夹路径 \
--save_dir # 保存检测结果的目录 \
--gpu # 是否使用GPU进行推理，如果不指定，则默认使用CPU \
--model_dir # 保存的模型文件的目录
```

## 输出结果
模型的输出结果讲保存在指定的save_dir中，每个dicom文件的测量结果会产生三个文件
```
├── save_dir
│   ├── dicom_1
│   │   ├── bvr_zei_image.png
│   │   ├── ca_image.png
│   │   ├── ei_image.png
│   │   ├── results.json
│   ├── dicom_2
│   │   ├── bvr_zei_image.png
│   │   ├── ca_image.png
│   │   ├── ...
```
其中“bvr_zei_image.png”， "ca_image.png"以及"ei_image.png"分别为各指标所测量的截面。

results.json的结构如下：
```
{
    "BVR": {
        "data": 0.8461538461538461, # BVR指数
        "line_1": [[111, 52], [111, 91]], # 表示侧脑室高度的两点
        "line_2": [[111, 52], [111, 19]]  # 表示侧脑室最高点到颅骨的两点
        },
    "zEI": {
        "data": 0.3482142857142857, # zEvans指数
        "line_1": [[111, 52], [111, 91]], # 表示侧脑室高度的两点
        "line_2": [[125, 131], [125, 19]] # 表示颅内高度的两点
        },
    "CA": {
        "data": 130.57901252510226, # CA指数
        "points": [[114, 51], [131, 58], [150, 48]] # 用于测量CA角度的三个点，左端点，中心点以及右端点
    },
    "EI": {
        "data": 0.35428571428571426, # Evans指数
        "line_1": [[205, 217], [30, 217]], # 颅骨内板最大宽度的两点
        "line_2": [[159, 106], [97, 106]]  # 侧脑室前角最大宽度的两点
    }
}
```
