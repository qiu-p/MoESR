<div align="center">

# MoE-SR


<div>
    <a href="https://github.com/qiu-p/MoESR"><img src=https://img.shields.io/badge/github-MoESR-red.svg alt="MoE-SR CI"></a>
</div>
<br>

</div>

The model (MoE-SR) is designed as an analogy to the structure of the Hybrid Expert Model (MoE) in the field of natural language processing. MoE-SR consists of a gating network and multiple expert networks, as well as a hybrid output module. The gating network primarily segments LR images based on their constituent categories in order to assign them to respective expert networks. The expert networks are responsible for upsampling the segmented portions allocated by the gating network to obtain high-resolution sub-images. Finally, the hybrid output module utilizes the Alpha Blending algorithm to stitch the sub-images together to generate the final high-resolution image. After balancing the trade-off between the inference speed and performance of MoE-SR, we modified Yolo to obtain the gating network.

MoE-SR can be regarded as a generic architecture for improving the performance of existing SR models. Therefore, for the expert networks, we adopt the form of network containers, which means existing SR models can be used for replacement. To fully leverage the performance of MoE-SR, we establish an HR-LR paired training set with additional category labels to train MoE-SR. Furthermore, through additional comparative experiments, when different SR models are used as expert networks, we can observe consistent specialization in handling images with corresponding category labels.


## Requirement
Python >= 3.7 (Recommend to use Anaconda or Miniconda)
PyTorch >= 1.7
NVIDIA GPU + CUDA
Linux (We have not tested on Windows)

## Usage
```
python test_pic.py --opt VDSR/tools/process_img2/options/option_set_03_vdsrwithsegJiekou/options.yml 
```

## Dataset
[专家网络模型训练集](https://pan.baidu.com/s/14KHhyPavRHmEY4eNhnkanA?pwd=73e9)

[对比实验测试集](https://pan.baidu.com/s/1clSU_kEWRDsSqnI_kVmXcw?pwd=3y02)

[MoESR 模型测试集](https://pan.baidu.com/s/1WsExE3PuGYgWKacGPo2vbA?pwd=k60j)

## Contact



