#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
//#include <hector_pose_estimation/filter/ekf.h>

int main(int argc, const char* argv[]) {
      if (argc != 1) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
      }


      torch::jit::script::Module module;
      try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("/home/r/Documents/FinalProject/FullUnit_1920_RogerMilroy/Code/hybrid_inference/src/torchscript/ekhi_model.pt");
      }
      catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
      }

      std::vector<torch::jit::IValue> inputs;
      try {

        inputs.emplace_back(torch::ones({1,  100, 2}));
        inputs.emplace_back(torch::ones({100, 4, 4}));


      } catch (const c10::Error& e) {
        std::cerr << "error creating inputs\n";
        return -1;
      }


    torch::Tensor output;

    output = module.forward(inputs).toTensor();

    std::cout << output.slice(/*dim=*/0, /*start=*/0, /*end=*/1) << '\n';

    std::cout << "ok\n";
}