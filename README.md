# FullUnit_1920_RogerMilroy

# This is the repository that contains all of the code and reports for the Final Year Project of Roger Milroy.

---


## Setup
### To set up the ROS sections of the project please follow the following instructions.

First you need a version of Ubuntu, preferably 18.04 LTS.

Make sure everything is up to date.

    run sudo apt update

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

#### ROS section

Follow wiki.ros.org/melodic/Installation/Ubuntu instructions for installing.
##### Use ros-melodic-desktop-full and follow all steps.

Install hector quadrotor packages.

Install qt4 which is a dependency of the hector project.

    sudo apt install qt4-default
    sudo apt install ros-melodic-geographic-msgs

Initialise the workspace and setup dependancies

    cd <repo_root>/Code/catkin_ws
    git submodule init
    git submodule update

    cd src

    git clone https://github.com/tu-darmstadt-ros-pkg/hector_models.git
    git clone https://github.com/tu-darmstadt-ros-pkg/hector_quadrotor.git
    git clone https://github.com/tu-darmstadt-ros-pkg/hector_localization.git
    git clone https://github.com/tu-darmstadt-ros-pkg/hector_slam.git
    git clone https://github.com/tu-darmstadt-ros-pkg/hector_gazebo.git

    cd ..

    catkin_make

It is highly likely that you will have to run catkin_make many times.
For some reason it builds in a bad order. Main thing is if it stalls at one point for more than say 5 reruns of catkin_make. You should check if you have run out of memory. Have a htop instance open.

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
