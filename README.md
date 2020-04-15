# FullUnit_1920_RogerMilroy

# This is the repository that contains all of the code and reports for the Final Year Project of Roger Milroy.

---


## Setup
### To set up the ROS sections of the project please follow the following instructions.

First you need a version of Ubuntu, preferably 18.04 LTS.

Make sure everything is up to date.

    sudo apt update

#### Install pytorch
(you will need python 3.6 +)

    pip3 install torch  
    
You might use pip but this depends on your system setup.
You may also need to use something different depending on your system. Please Google to resolve any issues you have with this step.

#### Install hybrid_inference

You will need to install the package in order for imports to work correctly.
Feel free to do this inside a virtualenv to keep your system tidier.

    cd <repo_root>/Code/hybrid_inference
    pip3 install --user .

### AWS OpenVPN Setup

For running on AWS with pose_estimation being remote
On AWS machine 
    
    cd ~
    wget https://git.io/vpn -O openvpn-install.sh
    chmod +x openvpn-install.sh
    sudo ./openvpn-install.sh
    sudo lsof -i

Select the defaults for almost everything in the script except for DNS, use 1.1.1.1.   
If you see openvpn near the bottom of the output of lsof -i then it was successful

#### ROS

##### INSTALL ROS
Follow wiki.ros.org/melodic/Installation/Ubuntu instructions for installing.
##### Use ros-melodic-desktop-full and follow all steps.

Install hector quadrotor packages.

Install qt4 and geographic msgs which are a dependencies of the hector project.

    sudo apt install qt4-default ros-melodic-geographic-msgs

Initialise the workspace and setup dependencies (you will need to change the start of the sed command to the path on your machine to this repo)

    cd <repo_root>/Code/catkin_ws/src

    git clone https://github.com/tu-darmstadt-ros-pkg/hector_models.git
    git clone https://github.com/tu-darmstadt-ros-pkg/hector_quadrotor.git
    git clone https://github.com/tu-darmstadt-ros-pkg/hector_slam.git
    git clone https://github.com/tu-darmstadt-ros-pkg/hector_gazebo.git
    git clone https://github.com/rogermilroy/hector_localization.git
    
    sed -i 's/set(CMAKE_PREFIX_PATH \/home\/r\/Documents\/FinalProject\/FullUnit_1920_RogerMilroy\/Code\/libtorch)/set(CMAKE_PREFIX_PATH \/home\/ubuntu\/FullUnit_1920_RogerMilroy\/Code\/libtorch)/' hector_localization/hector_pose_estimation_core/CMakeLists.txt

    cd ../..
    
    wget https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.4.0.zip
    
    unzip libtorch-cxx11-abi-shared-with-deps-1.4.0.zip
    
    rm libtorch-cxx11-abi-shared-with-deps-1.4.0.zip

    cd catkin_ws
    
    catkin_make
    
Then source, use the appropriate one depending on whether you are using bash or zsh.

    source devel/setup.bash
    
    source devel/setup.zsh

---

## Directory Structure
```
FullUnit_1920_RogerMilroy
├── Code
│   ├── catkin_ws
│   │   └── src
│   │       ├── CMakeLists.txt
│   │       ├── hector_quadrotor
│   │       ├── navdata_msgs
│   │       └── tum_ardrone
│   └── hybrid_inference
│       ├── setup.py
│       ├── src
│       │   ├── __init__.py
│       │   ├── data
│       │   │   ├── __init__.py
│       │   │   ├── synthetic_position_dataloader.py
│       │   │   └── synthetic_position_dataset.py
│       │   ├── data_utils
│       │   │   ├── __init__.py
│       │   │   └── converters.py
│       │   ├── models
│       │   │   ├── __init__.py
│       │   │   ├── decoding_model.py
│       │   │   ├── encoding_model.py
│       │   │   ├── graphical_model.py
│       │   │   ├── graphical_nn_model.py
│       │   │   ├── hybrid_inference_model.py
│       │   │   └── linear_model.py
│       │   └── training
│       │       ├── __init__.py
│       │       ├── evaluate_model.py
│       │       └── train_hybrid_inference.py
│       └── test
│           ├── __init__.py
│           ├── data_utils
│           │   ├── __init__.py
│           │   └── test_converters.py
│           ├── datasets
│           │   ├── __init__.py
│           │   └── test_synthetic_position_data.py
│           ├── models
│           │   ├── __init__.py
│           │   ├── test_graphical_model.py
│           │   ├── test_hybrid_inference.py
│           │   └── test_linear_model.py
│           └── training
│               └── test_training.py
├── README.md
└── Reports
    ├── Applications
    │   ├── Applications.pdf
    │   └── Applications.tex
    ├── GRIN
    │   ├── GRIN.pdf
    │   └── GRIN.tex
    ├── InterimReport
    │   ├── InterimReport.pdf
    │   └── InterimReport.tex
    ├── KalmanFilters
    │   ├── Kalman_Filters.pdf
    │   └── Kalman_Filters.tex
    ├── ProjectPlan
    │   ├── Project_Plan.pdf
    │   └── Project_Plan.tex
    └── resources
        ├── final_project.bib
        ├── final_report.cls
        └── images
            ├── GraphicalKalmanModel.png
            ├── Term1GanttChart.png
            ├── Term1GanttChartv2.png
            ├── Term1Ganttv1.png
            ├── Term2GanttChart.png
            ├── Term2Ganttv1.png
            ├── hybrid-inference-package-uml.png
            ├── hybrid-inference-package.uxf
            ├── hybrid-inference-uml.png
            ├── hybrid-inference.uxf
            └── logo.pdf
```

---
