# FullUnit_1920_RogerMilroy

# This is the repository that contains all of the code and reports for the Final Year Project of Roger Milroy.

---


## Setup
### To set up the ROS sections of the project please follow the following instructions.

First you need a version of Ubuntu, preferably 18.04 LTS.

Make sure everything is up to date.

    sudo apt update
    sudo apt upgrade

#### Install pytorch
(you will need python 3.6 +)

    pip3 install torch  
    
You might use pip but this depends on your system setup.
You may also need to use something different depending on your system. Please Google to resolve any issues you have with this step.

#### Install hybrid_inference

You will need to install the package in order for imports to work correctly.
Feel free to do this inside a virtualenv to keep your system tidier. If you have any issues 
with module import errors please uninstall and use a venv.

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

Set up ROS environment variables

    export ROS_IP=10.8.0.1
    export ROS_MASTER_URI=http://10.8.0.2:11311
    
Then on local VM 
    
    cd ~
    scp -i some_key.pem ubuntu@<aws-instance-public-ip>:~/client.ovpn .
    export ROS_IP=10.8.0.2
    sudo openvpn client.ovpn
    
Open a new terminal and run

    ifconfig

the bottom should show tun0 and ip as 10.8.0.2

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
│   │   ├── src
│   │   │   ├── CMakeLists.txt -> /opt/ros/melodic/share/catkin/cmake/toplevel.cmake
│   │   │   ├── navdata_msgs
│   │   │   ├── navigation
│   │   │   │   ├── CMakeLists.txt
│   │   │   │   ├── action
│   │   │   │   │   └── NavigateToPoint.action
│   │   │   │   ├── launch
│   │   │   │   │   └── empty_drone.launch
│   │   │   │   ├── package.xml
│   │   │   │   └── src
│   │   │   │       ├── controller.py
│   │   │   │       ├── fly_figure.py
│   │   │   │       ├── navigate_to_point.py
│   │   │   │       ├── publish_vel.py
│   │   │   │       ├── record_data.py
│   │   │   │       ├── talker.cpp
│   │   │   │       ├── test_load_model.cpp
│   │   │   │       ├── test_manage_vectors.cpp
│   │   │   │       └── utils.py
│   │   │   └── tum_ardrone
│   │   └── start_teleop.sh
│   └── hybrid_inference
│       ├── setup.py
│       ├── src
│       │   ├── __init__.py
│       │   ├── data
│       │   │   ├── __init__.py
│       │   │   ├── gazebo_dataloader.py
│       │   │   ├── gazebo_dataset.py
│       │   │   ├── synthetic_input_dataset.py
│       │   │   ├── synthetic_position_dataloader.py
│       │   │   └── synthetic_position_dataset.py
│       │   ├── data_utils
│       │   │   ├── __init__.py
│       │   │   └── converters.py
│       │   ├── models
│       │   │   ├── __init__.py
│       │   │   ├── batch_gru.py
│       │   │   ├── decoding_model.py
│       │   │   ├── encoding_model.py
│       │   │   ├── graphical_model.py
│       │   │   ├── graphical_nn_model.py
│       │   │   ├── hybrid_inference_model.py
│       │   │   ├── linear_input_model.py
│       │   │   ├── linear_model.py
│       │   │   ├── predictor.py
│       │   │   └── smoother.py
│       │   ├── torchscript
│       │   │   ├── batch_gru.py
│       │   │   ├── compile_to_torchscript.py
│       │   │   ├── graphical_models.py
│       │   │   ├── graphical_nn_model.py
│       │   │   ├── hybrid_inference_models.py
│       │   │   └── process_samples.py
│       │   ├── training
│       │   │   ├── __init__.py
│       │   │   ├── evaluate_extended.py
│       │   │   ├── evaluate_model.py
│       │   │   ├── loss.py
│       │   │   ├── train_extended_hybrid_inference.py
│       │   │   └── train_hybrid_inference.py
│       │   └── utils
│       │       ├── __init__.py
│       │       └── data_converters.py
│       └── test
│           ├── __init__.py
│           ├── data_utils
│           │   ├── __init__.py
│           │   └── test_converters.py
│           ├── datasets
│           │   ├── __init__.py
│           │   ├── test_gazebo_dataset.py
│           │   └── test_synthetic_position_data.py
│           ├── models
│           │   ├── __init__.py
│           │   ├── __pycache__
│           │   │   └── test_linear_input_model.cpython-37.pyc
│           │   ├── test_graphical_model.py
│           │   ├── test_hybrid_inference.py
│           │   ├── test_linear_input_model.py
│           │   └── test_linear_model.py
│           └── training
│               ├── __init__.py
│               ├── fixed_len10start1000_seq100.txt
│               ├── test_training.py
│               └── test_training_input.py
├── README.md
└── Reports
    ├── Applications
    │   ├── Applications.pdf
    │   └── Applications.tex
    ├── FinalReport
    │   ├── FinalReport.pdf
    │   └── FinalReport.tex
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
            ├── Navigation.uxf
            ├── NavigationUML.png
            ├── PredictionKalman.png
            ├── Term1GanttChart.png
            ├── Term1GanttChartv2.png
            ├── Term1Ganttv1.png
            ├── Term2GanttChart.png
            ├── Term2Ganttv1.png
            ├── TrelloScreenshot.png
            ├── ekhi_integration.uxf
            ├── ekhi_integrationUML.png
            ├── hybrid-inference-package-uml.png
            ├── hybrid-inference-package.uxf
            ├── hybrid-inference-uml.png
            ├── hybrid-inference.uxf
            ├── logo.pdf
            └── training2000.png
```

---
