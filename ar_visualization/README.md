
## Name
2D Depth Value "AR" Demonstration

## Description
The example scene used for the demo here is the Scene12_traj2_2 from HAMMER Dataset

## Visuals
![](output/depth_check.gif)

## Installation
Libraries "PIL.Image" and "cv2" need to be installed to your environment additionally to the built-in Python methods used 
Make sure "manydepth" environment is activated by "conda activate manydepth" before running.

## Usage
Basically getting the required images (depth_gt, depth_prediction, rgb_img and object_mask) in the data folder for an another scene is sufficient. 
This can be done easily by copying the images saved in "pointcloud/data/images" after the inference.
Afterwards the paths are be defined at the start of the main script for the new scene.

It is important to have the depth images in "uint8" format!

By running the main script you should be able to get a new gif with the predefined one directional trajectory. Some additional adjustments might be needed to get a reasonable gif. See the TODO's in the code.

 - python3 main.py

The resulting gif can be opened with the in-built ubuntu image viewers. 

## Support
For questions refer to "ge64jaq@tum.de"


## Contributing
Introducing transparency by adding alpha values(RGB-A image) would be possible.


## Authors and acknowledgment
Thanks to Patrick Ruhkamp, Witold Pacholarz, Kagan Kücükaytekin, Tobias Preintner and Volkan Tatlikazan.