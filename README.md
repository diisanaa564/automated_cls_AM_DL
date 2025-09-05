# Automated Classification of 3D Printed Defects Using Deep Learning

This repository contains Python implementations for FDM defect classification using deep learning.  
Models included: Basic CNN, ResNet34, Xception, and YOLOv11s-cls.  

## Repository Structure
- **models**: Model training scripts for CNN, ResNet34, Xception, and YOLOv11s.
- **dataset construction**: Dataset preparation and augmentation scripts.
- **results**: Training logs, performance metrics (JSON), and plots of accuracy/loss curves.
- **README.md**: Project description and references.
- 
## Dataset  
The dataset used in this project is a merged version of six sources.  
You can access the combined dataset here:  
[Combined Dataset Link](https://computingservices-my.sharepoint.com/:f:/g/personal/dn564_bath_ac_uk/EiZLKjPEwhhHgqQYKu3tGeMBxgHWY13sOilsYN2bp-XZhQ?e=7RTzPB)  

### Original Sources :
No defect:
https://universe.roboflow.com/arizonastateuniversity/3d-printing-cm/browse?queryText=class%3AOK&pageSize=50&startingIndex=0&browseQuery=true; 
https://www.kaggle.com/datasets/bshaurya/3d-printing-success?select=success+%28111%29.jpg; 
Defects:
https://universe.roboflow.com/new-dataset-fbe1u/combined-extrusion/dataset/3; 
https://app.roboflow.com/multimodal-inspection-additive-manufacturing/under-extrusion-pf0uy-dewzb/1; 
https://app.roboflow.com/multimodal-inspection-additive-manufacturing/warping-3d-prints-kxwl0/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true; 
https://universe.roboflow.com/automatisierung-projekt-aajut/3d-ho5at/dataset/4

## Citation
If you use this repository, please cite:  
*Nabila, D. Automated Classification of 3D Printed Defects Using Deep Learning, 2025.- University of Bath*
