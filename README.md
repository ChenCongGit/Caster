# Caster
该项目模仿自然场景文本识别网络ASTER([here](https://ieeexplore.ieee.org/abstract/document/8395027/))的Tensorflow实现

ASTER是2018年提出的识别精度较好的行文本识别网络，在多个公共数据集上都拥有非常出色的精度。
![](/相关图片/ASTER模型.png "ASTER模型")

## 引用
```
@article{bshi2018aster,
  author  = {Baoguang Shi and
               Mingkun Yang and
               Xinggang Wang and
               Pengyuan Lyu and
               Cong Yao and
               Xiang Bai},
  title   = {ASTER: An Attentional Scene Text Recognizer with Flexible Rectification},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  volume  = {}, 
  number  = {}, 
  pages   = {1-1},
  year    = {2018}, 
}

@inproceedings{ShiWLYB16,
  author    = {Baoguang Shi and
               Xinggang Wang and
               Pengyuan Lyu and
               Cong Yao and
               Xiang Bai},
  title     = {Robust Scene Text Recognition with Automatic Rectification},
  booktitle = {2016 {IEEE} Conference on Computer Vision and Pattern Recognition,
               {CVPR} 2016, Las Vegas, NV, USA, June 27-30, 2016},
  pages     = {4168--4176},
  year      = {2016}
}

If you find this project helpful to your research, please quote the above paper and the following license:
```

# 创建ASTER运行环境
Denote the root directory path of Caster by `${Caster_root}`. 
Create and activate the required virtual environment by:
```
conda env create -f `${Caster_root}/Caster_conda_env.yml`
```

## 初始化配置
Denote the root directory path of Caster by `${Caster_root}`. 
1. Add the path of `${Caster_root}` to your `PYTHONPATH`:
```
export PYTHONPATH="$PYTHONPATH:${Caster_root}"
```
2. Go to `c_ops/` and run `build.sh` to build the custom operators
```
cd `${Caster_root}/c_ops`
bash build.sh
```
3. Build the protobuf files
```
protoc Caster/protos/*.proto --python_out=.
```

## 运行程序
通过以下命令进行训练、测试、推理，其中训练过程的相关参数在`${Caster_root}/experiments/config/trainval.prototxt`文件中配置，你可以通过修改该文件配置自己的训练过程
1. 训练
```
CUDA_VISIBLE_DEVICES=0,1 python tarin.py --num_clones=1 --dataset_dir=${Caster_root}datasets/ --exp_dir=${Caster_root}/experiments/
```
2. 测试评估
```
CUDA_VISIBLE_DEVICES=0,1 python eval.py --repeat=True --dataset_dir=${Caster_root}datasets/ --exp_dir=${Caster_root}/experiments/
```
3. 推理
```
CUDA_VISIBLE_DEVICES=0,1 python demo.py --input_image_dir=${Caster_root}datasets/ --exp_dir=${Caster_root}/experiments/ --check_num=200000
```
