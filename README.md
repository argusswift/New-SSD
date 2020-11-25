# New-SSD
High precision and fast speed for SSD

| System | VOC2007 test |VOC2012 test | **FPS** (TitanX) | #Boxes | Input resolution
|:-------|:-----:|:-----:|:-------:|:-------:|:-------:|
| SSD300 (VGG16) | 77.2 | 75.8 | 46 | 8732 | 300 x 300 |
| SSD512 (VGG16) | 79.8 | 78.5 | 19 | 24564 | 512 x 512 |
| SSD300 (VGG16)(this repo) | 80.0 | 77.5 | - | 8732 | 300 x 300 |
| SSD321 (ResNet101)(this repo) | 79.5 | 76.4 | 27.5 | 10325 | 321 x 321 |
| SSD512 (VGG16)(this repo) | 81.6 | 80.0| -  | 24564 | 512 x 512 |
| SSD513 (ResNet101)(this repo) | **83.0** | **81.3** | 16 | 25844 | 513 x 513 |

## Dependencies
Library: OpenCV-Python, PyTorch>0.4.0, Ubuntu 14.04

## Dataset
### PascalVOC
  ```Shell
  # Download the data.
  cd $HOME/data
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  # Extract the data.
  tar -xvf VOCtrainval_11-May-2012.tar
  tar -xvf VOCtrainval_06-Nov-2007.tar
  tar -xvf VOCtest_06-Nov-2007.tar
  ```
### MSCOCO 2017 ([download link](http://cocodataset.org/#download))
  ```Shell
	#step1: download the following data and annotation
	2017 Train images [118K/18GB]
	2017 Val images [5K/1GB]
	2017 Test images [41K/6GB]
	2017 Train/Val annotations [241MB]
	#step2: arrange the data to the following structure
	COCO
 	---train
	---test
	---val
	---annotations
  ```


## Train/Test/Evaluation
```Shell
1. Change the mode in main.py
2. Change parameters such as root (data directory) in config/ssd_config.py
3. python main.py
```
