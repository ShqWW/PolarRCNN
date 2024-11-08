# Install
The user is advised to ready the foundational environments with Pytorch (with CUDA). We suggest the following versions:
```
python>=3.10
pytorch>=2.0.0
```
Subsequently, execute the following commands to install extra packages:
```
pip install -r requirements.txt
cd ops/NMSOPS
python setup.py install
cd -
```

# Data Preprocessing

Before executing the framework, dataset should be prepared. This framework accommodates five dataset benchmark including CULane, LLAMAS, TuSimple, Curvelanes and DL-Rail.

In CULane dataset, we exclude some trainng set. The new training list could be generated by [exclude_culane.py](./exclude_culane.py) or alternatively, the new training list [train_gt_new.txt](https://github.com/ShqWW/PolarRCNN/releases/download/v0.0/train_gt_new.txt) could be directly download.

The label of LLAMAS need to be preprocessed before training, thereby causes the initialization  of training time-consuming. Don't worry.


The arrangement of the dataset ought to be as delineated below:
### CULane
```
CULane/
|-- driver_23_30frame
|-- driver_37_30frame
...... (more directories)
|-- driver_193_90frame
|-- list/
|   |-- test_split
|   |-- test.txt
|   |-- train_gt.txt
|   |-- train_gt_new.txt
|   |-- train.txt
|   |-- val_gt.txt
```

### LLAMAS
```
LLAMAS/
|-- color_images
|   |-- test
|   |-- train
|   |-- valid
|-- labels
|   |-- train
|   |-- valid
```

### TuSimple 
```
TuSimple/
|-- train_set
|   |-- clips
|   |-- seg_label
|   |-- label_data_0313.json
|   |-- label_data_0531.json
|   |-- label_data_0601.json
|-- test_set
|   |-- clips
|   |-- test_tasks_0627.json
|-- test_label.json
```


### Curvelanes
```
Curvelanes/
|-- train
|   |-- images
|   |-- labels
|   |-- train.txt
|-- valid
|   |-- images
|   |-- labels
|   |-- valid.txt
|-- test
|   |-- images
```
### DL-Rail
```
DL-Rail/
|-- list
|   |-- train.txt
|   |-- train_gt.txt
|   |-- test.txt
|-- videos
|   |-- video-000
|   |-- video-001
...... (more directories)
|   |-- video-049
```



# Run

### Train

The template for the command to initiate single GPU training is as follows:
```
python -u train.py --gpu_no ${GPUNO} --save_path ${WEIGHTPATH} --cfg ${CONFIGPATH} --iter_display ${ITERDISPLAY}
```
`${GPUNO} `: GPU Number;

`${WEIGHTPATH} `: Directory for the preservation of model weights;

`${CONFIGPATH} `: Location of the configuration file.

`${ITERDISPLAY} `: The iteration interval to print the loss.

#### Example:
```
python -u train.py --gpu_no 0 --save_path ./ckpt_culane_r18 --cfg ./Config/polarrcnn_culane_r18.py --iter_display 5
```

You can also initiate the training with multiple GPUs using the `torchrun` command:
```
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 --master_port=12345 train.py --save_path ./ckpt_culane_r18 --cfg ./Config/polarrcnn_culane_r18.py --iter_display 5 --is_multigpu 1
```
In multi-GPU training, the `batch_size` parameter parameter within the configuration file signifies the total batch size rather than the batch size per GPU, and it must be evenly divisible by the total number of used GPUs.

### Evaluation & Visualization
The template for the command for evaluation or visualization is:
```
python -u test.py --gpu_no ${GPUNO} --weight_path ${WEIGHTPATH} --result_path ${RESULTPATH} --cfg ${CONFIGPATH} --is_view ${VIEWFLAG} --view_path ${VIEWPATH}
```
`${GPUNO} `: GPU Number;

`${WEIGHTPATH} `: Directory for the preservation of model weights;

`${RESULTPATH} `: Directory for the preservation of detection results;

`${CONFIGPATH} `: Location of the configuration file;

`${VIEWFLAG} `: `True` for visualization and `False` (default) for evaluation;

`${VIEWPATH} `: Directory for the preservation of visualization results.

#### Example:

To evaluate the model performance on test set, execute with:
```
python -u test.py  --gpu_no 0 --weight_path ./weight_path/polarrcnn_culane_r18.pth --result_path ./result_culane --cfg ./Config/polarrcnn_culane_r18.py
```

or you can also evaluate the model performance on validation set:
```
python -u test.py  --gpu_no 0 --weight_path ./weight_path/polarrcnn_culane_r18.pth --result_path ./result_culane --cfg ./Config/polarrcnn_culane_r18.py --is_val 1
```

To view the visualization of the detection result, execute with:
```
python -u test.py  --gpu_no 0 --weight_path ./weight_path/polarrcnn_culane_r18.pth --result_path ./result_culane --cfg ./Config/polarrcnn_culane_r18.py --is_view 1 --view_path ./view_culane
```



