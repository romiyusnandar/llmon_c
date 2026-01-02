#include <iostream>
#include "../include/tensor.h"
#include "../include/ops.h"
#include "../include/optimizer.h"

int main() {
    std::cout << "=== Micro-LLM C++ ===\n\n";

    // use simple dataset, data points batch size 3, dim 1
    auto input = Tensor::create(3, 1);
    input->data = {1.0f, 2.0f, 3.0f};

    // we want the model learn the function of y = 2x
    auto target = Tensor::create(3, 1);
    target->data = {2.0f, 4.0f, 6.0f};

    /**
     * Model.
     * One unbiased linear layer: y = x * w
     * Since the input is (3x1) and we want to process each
     * row independently...
     * Wait, to simplify the matrix:
     * Let's assume the input is (3 rows x 1 feature).
     * So we need the weight (1 feature x 1 output)
     */
    auto weights = Tensor::create(1, 1);
    weights->random_init();

    std::cout << "Initial Weight: " << weights->data[0] << "\n\n";

    /**
     * Optimizer.
     * We use learning rate 0.1 for faster convergence
     */
    SGD optimizer({weights}, 0.1f);

    // Training loop
    std::cout << "Start Training (100 Epoch)...\n";
    for (int epoch = 0; epoch <= 100; epoch++) {

        // Forward pass
        // Input (3x1) @ Weights (1x1) -> Output (3x1)
        auto pred = matmul(input, weights);
        auto loss = mse_loss(pred, target);
        optimizer.zero_grad();
        loss->backward();
        optimizer.step();

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch
                      << " | Loss: " << loss->data[0]
                      << " | Weight: " << weights->data[0] << "\n";
        }
    }

    std::cout << "\n=== Final Result ===\n";
    std::cout << "Target Weight: 2.0000\n";
    std::cout << "Model Weight : " << weights->data[0] << "\n";

    // Test prediction
    float test_val = 5.0f;
    std::cout << "Test input 5.0 -> Pediction: " << test_val * weights->data[0] << "\n";

    return 0;
}