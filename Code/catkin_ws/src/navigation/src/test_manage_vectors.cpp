//
// Created by r on 3/23/20.
//
#include <torch/script.h> // One-stop header.

#include <iostream>
#include <hector_pose_estimation/filter/ekf.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <hector_pose_estimation/state.h>


int main(int argc, const char *argv[]) {
    if (argc != 1) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    hector_pose_estimation::State::Vector test_y;
    test_y = hector_pose_estimation::State::Vector(10);

    hector_pose_estimation::State::SystemMatrix test_matrix = hector_pose_estimation::State::SystemMatrix(10, 10);

    std::vector<std::vector<float_t>> some_ys;

    auto tmp = std::vector<float_t>{1.0, 2.0};

    some_ys.push_back(tmp);
    some_ys.push_back(tmp);

    auto eig_vec = hector_pose_estimation::State::columnVectorToVector(test_y);


    // This section converts a SystemMatrix to a tensor representation of it.
    auto eig_mat = hector_pose_estimation::State::systemMatrixToVector(test_matrix);

    // Converts each sub vector to tensor and stacks into a container vector
    std::vector<at::Tensor> tensors;
    for (const auto& x : eig_mat) {
        std::cout << x << std::endl;
        std::cout  << x.size() << std::endl;
        tensors.push_back(torch::tensor(x));
    }
    // This stacks those tensors to make a 2d tensor. Must be transposed
    // due to mapping taking columnwise and stack going rowwise.
    at::Tensor t_tensors = torch::stack(tensors).t();

    // Print.
    std::cout << "Original Eig: \n" << test_matrix << std::endl;

    std::cout << " some spacer " << std::endl;

    std::cout << t_tensors << std::endl;


    auto tens_y = torch::tensor(tmp);
    auto tens_x = torch::tensor(tmp);

    auto tens_xy = torch::stack({tens_x, tens_y});

//    std::cout << tens_y << std::endl;


//    std::cout << tens_xy << std::endl;


}
