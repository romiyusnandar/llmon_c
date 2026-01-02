#include <iostream>
#include "../include/tensor.h"
#include "../include/ops.h"

int main() {
    std::cout << "=== Micro-LLM C++ Engine ===\n\n";

    // 1. Setup Input
    Tensor input(1, 5);
    input.data = {-1.0f, 0.5f, 2.0f, -3.0f, 1.5f};
    std::cout << "[Input]\n";
    input.print();

    // 2. Setup Weights
    Tensor weights(5, 3);
    weights.random_init();
    std::cout << "[Weights]\n";
    weights.print();

    // 3. Layer 1: Linear (MatMul)
    Tensor hidden = matmul(input, weights);
    std::cout << "[Hidden Layer (Pre-ReLU)]\n";
    hidden.print();

    // 4. Activation: ReLU
    Tensor output = relu(hidden);
    std::cout << "[Output (Post-ReLU)]\n";
    output.print();
    std::cout << "Note: The negative values ​​above should be 0.0000\n";

    return 0;
}