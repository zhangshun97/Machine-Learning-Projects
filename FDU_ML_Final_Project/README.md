# Painting Authentication and Style Transfer
This is the final project of machine learning course @SDS, Fudan Univ.

The project is co-finished with [Shihan Ran](https://github.com/Rshcaroline) and [Jiancong Gao](https://github.com/jcgao).

#### 1 Authentication

- Our result:

![image](https://github.com/zhangshun97/Machine-Learning-Projects/blob/master/images/authentication.png)

*More details please refer to [our paper]().*

#### 2 Style Transfer

- Getting Started
  - requirements

    - Python 2.7
    - PyTorch 0.3.0
    - PIL

  - Run demo: 

    - `python neural_style_transfer.py`

  - Transfer any style image `style_img_path` to any content image `content_img_path`:

    - ``python neural_style_transfer.py -S style_img_path -C content_img_path`
    - run ``python neural_style_transfer.py --help` for more arguments

    *Any JPG/PNG image with resolution more than `512,512`(GPU) or `128,128`(cpu) shall work.*

- Model Overview

![image](https://github.com/zhangshun97/Machine-Learning-Projects/blob/master/images/styleTransfer.png)

- Result Highlights

![image](https://github.com/zhangshun97/Machine-Learning-Projects/blob/master/images/styleTransfer2.png)

*More details please refer to [our paper]().*

## 