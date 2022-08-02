# Depth from Polarization
***

## 3D Pointcloud Demonstration

### Description
The example scene used for the pointcloud here is defined in the 
"HAMMER_pointcloud/test_files.txt" file. 
The possible scenes from HAMMER Dataset are:
 - scene12_traj2_2
 - scene13_traj2_1
 - scene14_traj2_1

Use one scene only per run to ensure you get the pointcloud for that scene only.
 
### Installation
Libraries "PIL.Image" and "open3d" need to be installed to your environment additionally to the built-in Python methods used.
Make sure "manydepth" environment is activated by "conda activate manydepth" before running.

### Usage
Running the script by the following commands in terminal allows to see the pointclouds for each architecture version for
predicted and ground truth depth (first one is RGB only) :

 - python3 eval_pointcloud_main.py 
 - python3 eval_pointcloud_main.py --use_xolp True
 - python3 eval_pointcloud_main.py --use_normals True
 - python3 eval_pointcloud_main.py --use_xolp True --use_normals True 


 - see the TODO's in the "eval_pointcloud.py" for further details

### Support
For questions refer to "ge64jaq@tum.de"

### Contributing
Training a network solely based on "nearest" interpolation for image scaling may help reduce the pixels at object boundaries.

***

## 2D Depth Value "AR" Demonstration

### Description
The example scene used for the demo here is the Scene12_traj2_2 from HAMMER Dataset

### Visuals
![](ar_visualization/output/depth_check.gif)

### Installation
Libraries "PIL.Image" and "cv2" need to be installed to your environment additionally to the built-in Python methods used 
Make sure "manydepth" environment is activated by "conda activate manydepth" before running.

### Usage
Basically getting the required images (depth_gt, depth_prediction, rgb_img and object_mask) in the data folder for an another scene is sufficient. 
This can be done easily by copying the images saved in "pointcloud/data/images" after the inference.
Afterwards the paths are be defined at the start of the main script for the new scene.

It is important to have the depth images in "uint8" format!

By running the main script you should be able to get a new gif with the predefined one directional trajectory. Some additional adjustments might be needed to get a reasonable gif. See the TODO's in the code.

 - python3 main.py

The resulting gif can be opened with the in-built ubuntu image viewers. 

### Support
For questions refer to "ge64jaq@tum.de"

***

## TEMPLATE 

## Introduction

todo: add a gif from demonstration, or an attractive image representing depth from polarization.

Welcome to our project repository! Since we have different versions of our architecture, 
we assigned a different branch for each version. To use a specific version, simply check 
out to the corresponding branch of the architecture.


## Branches

| Branch  | Architecture                         | Information                                                                                                                                                   |
|---------|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `main`    | Final architecture                   | The architecture we present as our final. This is the architecture we ran our ablation studies.                                                               |
| `attention` | Best quantitative results on objects | This architecture enhances the main architecture with use of attention after combining modalities                                                             |
| `separate_normals_decoder`| Use a decoder after normals encoder  | The decoder directly predicts normals. These normals are compared with normals calculated from ground truth to drive the supervised learning.                 |
| `architecture1+` | Intermediate architecture | This architecture contains deeper XOLP and normals encoders. There is no joint encoder, all features are concatenated and directly pushed into depth decoder. |

## Training
Here are training instructions

## Evaluation
Here are evaluation instructions

## Final Presentation
Maybe add final presentation slides here as well.

# The rest
the rest from here is the templated, old content. keeping it for now.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!
`
## Add your files
`
- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.lrz.de/at3dcv2022/depthfrompol.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.lrz.de/at3dcv2022/depthfrompol/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Automatically merge when pipeline succeeds](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!).  Thank you to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README
Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
