# AD/PD三个指标测量

本仓库主要用于zEvans, Evans, BVR, CA四个指标测量

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
--path_predictions ./results # 保存检测结果的目录 \
--gpu # 是否使用GPU进行推理，如果不指定，则默认使用CPU \
--model_dir # 保存的模型文件的目录
```