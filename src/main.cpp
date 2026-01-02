#include <iostream>
#include "../include/tensor.h"
#include "../include/ops.h"

int main() {
    std::cout << "=== Micro-LLM C++: Phase 2 (Autograd) ===\n\n";

    auto input = Tensor::create(1, 3);
    input->data = {1.0f, -2.0f, 3.0f};
    std::cout << "[Input]\n"; input->print();

    auto weights = Tensor::create(3, 1);
    weights->data = {0.5f, 0.5f, 0.5f};
    std::cout << "[Weights]\n"; weights->print();
    auto hidden = matmul(input, weights);
    auto output = relu(hidden);

    std::cout << "[Output Forward]\n";
    output->print();

    std::cout << "Running Backward Pass...\n";
    output->backward();

    std::cout << "\n[Gradients at Input]\n";
    for(int i=0; i<3; i++) {
        std::cout << input->grad[i] << " ";
    }
    std::cout << "\n(Expected: 0.5 0.5 0.5)\n";

    std::cout << "\n[Gradients at Weights]\n";
    for(int i=0; i<3; i++) {
        std::cout << weights->grad[i] << " ";
    }
    std::cout << "\n(Expected: 1.0 -2.0 3.0)\n";

    return 0;
}