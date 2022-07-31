
## Name
3D Pointcloud Demonstration

## Description
The example scene used for the pointcloud here is defined in the HAMMER_pointcloud/test_files.txt file. 
The possible scenes from HAMMER Dataset are:
 - scene12_traj2_2
 - scene13_traj2_1
 - scene14_traj2_1

Use one scene only per run to ensure you get the pointcloud for that scene only.
 
## Installation
Libraries "PIL.Image" and "open3d" need to be installed to your environment additionally to the built-in Python methods used.
Make sure "manydepth" environment is activated by "conda activate manydepth" before running.

## Usage
Running the script by the following commands in terminal allows to see the pointclouds for each architecture version for
predicted and ground truth depth (first one is RGB only) :

 - python3 eval_pointcloud_main.py 
 - python3 eval_pointcloud_main.py --use_xolp True
 - python3 eval_pointcloud_main.py --use_normals True
 - python3 eval_pointcloud_main.py --use_xolp True --use_normals True 


 - see the TODO's in the "eval_pointcloud.py" for further details

## Support
For questions refer to "ge64jaq@tum.de"

## Contributing
Traning a network solely based on "nearest" interpolation for image scaling may help reduce the pixels at object boundaries.

## Authors and acknowledgment
Thanks to Patrick Ruhkamp, Witold Pacholarz, Kagan Kücükaytekin, Tobias Preintner and Volkan Tatlikazan.
