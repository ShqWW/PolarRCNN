<div align="center">

# Polar R-CNN: A New Simple Baseline for 2D Lane Detection

</div>


<!-- <!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        
    </style>
</head> -->

<link rel="stylesheet" href="styles.css">


## Introduction

<div align="center">
  <img src="resources/model-architecture.png" style="width: 100%, height: auto;"/>
</div>

#PyTorch implementation of the paper "[Polar R-CNN: End-to-End Lane Detection with Fewer Anchors](https://arxiv.org/pdf/2411.01499)".

Features:
- Reduced Anchor Requirements: 20 anchors is all you need.
- End-to-End Capability: Facilitates training and evaluation absent NMS post-processing.
- Deployment-friendly: Comprises solely of CNNs and MLPs.
- Scalable Framework: Encompasses data preprocessing, model training, performance assessment, and visualization.


## Demo 


<table>
    <tr>
        <td><img src="resources/view_dataset/culane/pred.jpg" class=auto_img></td>
        <td><img src="resources/view_dataset/tusimple/pred.jpg" class=auto_img></td>
        <td><img src="resources/view_dataset/llamas/pred.jpg" class=auto_img></td>
        <td><img src="resources/view_dataset/dlrail/pred.jpg" class=auto_img></td>
    </tr>
    <tr>
        <td><img src="resources/view_dense/pred1.jpg" class=auto_img></td>
        <td><img src="resources/view_dense/pred2.jpg" class=auto_img></td>
        <td><img src="resources/view_dense/pred3.jpg" class=auto_img></td>
        <td><img src="resources/view_dense/pred4.jpg" class=auto_img></td>
    </tr>
</table>

## Get started
For the preparation of datasets and environments, as well as detailed commands, please refer to [INSTALL.md](./INSTALL.md).


## Trained Weights
We provide trained model weights and corresponding config files for CULane, Tusimple, LLAMAS, DL-Rail, and CurveLanes.

| Dataset    | Backbone | Performance (NMS-free) | Config | Weight-Link |
| :--------: | :------: | :-----------: | :----: | :---------: |
| CULane     | ResNet18 |    80.81 (F1@50)       | [culane_r18](Config/polarrcnn_culane_r18.py) | [polarrcnn_culane_r18.pth](https://github.com/ShqWW/PolarRCNN/releases/download/v0.0/polarrcnn_culane_r18.pth) |
| CULane     | ResNet34 |    80.92 (F1@50)       | [culane_r34](Config/polarrcnn_culane_r34.py) | [polarrcnn_culane_r34.pth](https://github.com/ShqWW/PolarRCNN/releases/download/v0.0/polarrcnn_culane_r34.pth) |
| CULane     | ResNet50 |    81.34 (F1@50)       | [culane_r50](Config/polarrcnn_culane_r50.py) | [polarrcnn_culane_r50.pth](https://github.com/ShqWW/PolarRCNN/releases/download/v0.0/polarrcnn_culane_r50.pth) |
| CULane     | DLA34    |    81.49 (F1@50)       | [culane_dla34](Config/polarrcnn_culane_dla34.py) | [polarrcnn_culane_dla34.pth](https://github.com/ShqWW/PolarRCNN/releases/download/v0.0/polarrcnn_culane_dla34.pth) |
| Tusimple   | ResNet18 |    97.94 (F1)          | [tusimple_r18](Config/polarrcnn_tusimple_r18.py) | [polarrcnn_tusimple_r18.pth](https://github.com/ShqWW/PolarRCNN/releases/download/v0.0/polarrcnn_tusimple_r18.pth) |
| LLAMAS     | ResNet18 |    96.06 (F1@50)       | [llamas_r18](Config/polarrcnn_llamas_r18.py) | [polarrcnn_llamas_r18.pth](https://github.com/ShqWW/PolarRCNN/releases/download/v0.0/polarrcnn_llamas_r18.pth) |
| LLAMAS     | DLA34    |    96.14 (F1@50)       | [llamas_dla34](Config/polarrcnn_llamas_dla34.py) | [polarrcnn_llamas_dla34.pth](https://github.com/ShqWW/PolarRCNN/releases/download/v0.0/polarrcnn_llamas_dla34.pth) |
| DL-Rail    | ResNet18 |    97.00 (F1@50)       | [dlrail_r18](Config/polarrcnn_dlrail_r18.py) | [polarrcnn_dlrail_r18.pth](https://github.com/ShqWW/PolarRCNN/releases/download/v0.0/polarrcnn_dlrail_r18.pth) |
| CurveLanes | DLA34    |    87.29 (F1@50)       | [curvelanes_dla34](Config/polarrcnn_curvelanes_dla34.py) | [polarrcnn_curvelanes_dla34.pth](https://github.com/ShqWW/PolarRCNN/releases/download/v0.0/polarrcnn_curvelanes_dla34.pth) |


## Citation

```BibTeX
@inproceedings{zheng2022clrnet,
  title={Clrnet: Cross layer refinement network for lane detection},
  author={Zheng, Tu and Huang, Yifei and Liu, Yang and Tang, Wenjian and Yang, Zheng and Cai, Deng and He, Xiaofei},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={898--907},
  year={2022}
}

@inproceedings{honda2024clrernet,
  title={CLRerNet: improving confidence of lane detection with LaneIoU},
  author={Honda, Hiroto and Uchida, Yusuke},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1176--1185},
  year={2024}
}

@inproceedings{chen2024sketch,
  title={Sketch and Refine: Towards Fast and Accurate Lane Detection},
  author={Chen, Chao and Liu, Jie and Zhou, Chang and Tang, Jie and Wu, Gangshan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={2},
  pages={1001--1009},
  year={2024}
}
```
