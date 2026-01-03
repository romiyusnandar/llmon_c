/**
 * Demo: Comparing SGD vs Adam Optimizer
 * Shows how Adam converges faster than SGD
 */

#include <iostream>
#include "../include/tensor.h"
#include "../include/ops.h"
#include "../include/optimizer.h"

void train_with_sgd() {
    std::cout << "=== Training with SGD (LR=0.1) ===\n";

    auto input = Tensor::create(3, 1);
    input->data = {1.0f, 2.0f, 3.0f};

    auto target = Tensor::create(3, 1);
    target->data = {2.0f, 4.0f, 6.0f};

    auto weight = Tensor::create(1, 1);
    weight->random_init();

    SGD optimizer({weight}, 0.1f);

    for (int epoch = 0; epoch <= 100; epoch += 20) {
        auto pred = matmul(input, weight);
        auto loss = mse_loss(pred, target);

        optimizer.zero_grad();
        loss->backward();
        optimizer.step();

        std::cout << "Epoch " << epoch
                  << " | Loss: " << loss->data[0]
                  << " | Weight: " << weight->data[0] << "\n";
    }
}

void train_with_adam() {
    std::cout << "\n=== Training with Adam (LR=0.5) ===\n";

    auto input = Tensor::create(3, 1);
    input->data = {1.0f, 2.0f, 3.0f};

    auto target = Tensor::create(3, 1);
    target->data = {2.0f, 4.0f, 6.0f};

    auto weight = Tensor::create(1, 1);
    weight->random_init();

    // Adam with adaptive learning rate
    Adam optimizer({weight}, 0.5f);

    for (int epoch = 0; epoch <= 100; epoch += 20) {
        auto pred = matmul(input, weight);
        auto loss = mse_loss(pred, target);

        optimizer.zero_grad();
        loss->backward();
        optimizer.step();

        std::cout << "Epoch " << epoch
                  << " | Loss: " << loss->data[0]
                  << " | Weight: " << weight->data[0] << "\n";
    }
}

int main() {
    train_with_sgd();
    train_with_adam();

    std::cout << "\nBoth optimizers work, but Adam adapts learning rate automatically!\n";
    std::cout << "   For complex models with many parameters, Adam usually performs better.\n";
    return 0;
}
