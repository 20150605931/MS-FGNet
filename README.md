## Environment

- Pytorch 1.7.1+

- Python 3.7+

## Requirements

- torch==1.7.1
- torchvision==0.8.2
- scipy==1.4.1
- numpy==1.19.2
- matplotlib==3.2.1
- opencv_python==3.4.1.15
- tqdm==4.62.3
- Pillow==8.4.0
- h5py==3.1.0
- terminaltables==3.1.0
- packaging==21.3


## Run ccommand

After getting the DIA-MS images, put it in the datas/RGB folder, run the below command to crop unnecessary information from DIA-MS images, and then generate the initial preprocessed data in the datas/format folder

```markup
python tools/format_MSImg.py
```

Split the data into training and testing sets (alternatively, generate K-fold cross validation data, the default is 5 folds) in the datasets/ foder. 

```markup
python tools/split_data.py
```

Modify the label information in the datas/annotations.txt to match your dataset. Then, run the following command to generate the .txt files for training and testing sets. The generated datas/train.txt and datas/test.txt include the information of data paths and their corresponding ground truth labels. (alternatively, generate K-fold .txt files such as train1/test1.txt train2/test2.txt ... )

```markup
python tools/get_annotation.py
```

train/valid model

```markup
# Use the GPU to train the MS-FGNet model
CUDA_VISIBLE_DEVICES=1 python tools/train.py models/resnet/MS-FGNet.py --device cuda --kflod-validation -1
# Or use the K-fold cross validation to train the MS-FGNet model
CUDA_VISIBLE_DEVICES=1 python tools/train.py models/resnet/MS-FGNet.py --device cuda --kflod-validation 0


# Verify the MS-FGNet model. Note that you need to replace test=dict() in models/resnet/MS-FGNet.py with the trained weights
python tools/evaluation.py models/resnet/MS-FGNet.py --kflod-validation -1
# Or use the following command when performing K-fold cross-validation 
python tools/evaluation.py models/resnet/MS-FGNet.py --kflod-validation 0

# Visual CAM picture
python tools/vis_cam.py ./datasets/test/FA models/resnet/MS-FGNet.py --save-path ./logs/CAM_visualization/FA --target-category 0
```

# Data
If you need data, please contact the authors by email in time.


## Reference

    @repo{2020mmclassification,
        title={OpenMMLab's Image Classification Toolbox and Benchmark},
        author={MMClassification Contributors},
        howpublished = {\url{https://github.com/open-mmlab/mmclassification}},
        year={2020}
    }

    this repo is based on [repo](https://github.com/Fafa-DL/Awesome-Backbones)

