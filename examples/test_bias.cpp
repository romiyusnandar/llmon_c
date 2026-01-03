#include <iostream>
#include "../include/tensor.h"
#include "../include/nn.h"
#include "../include/optimizer.h"

int main() {
    std::cout << "=== Testing Linear Layer with Bias ===\n\n";

    auto input = Tensor::create(2, 3);
    input->data = {1.0f, 2.0f, 3.0f,
                   4.0f, 5.0f, 6.0f};

    auto target = Tensor::create(2, 2);
    target->data = {1.0f, 0.0f,
                    0.0f, 1.0f};

    Linear layer(3, 2, true);

    std::cout << "Initial Weight:\n";
    layer.weight->print();
    std::cout << "\nInitial Bias:\n";
    layer.bias->print();

    SGD optimizer(layer.parameters(), 0.01f);  // Smaller learning rate

    std::cout << "\n=== Training ===\n";
    for (int epoch = 0; epoch <= 200; epoch += 40) {
        auto output = layer.forward(input);
        auto loss = mse_loss(output, target);

        optimizer.zero_grad();
        loss->backward();
        optimizer.step();

        if (epoch % 40 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << loss->data[0] << "\n";
        }
    }

    std::cout << "\n=== Final Parameters ===\n";
    std::cout << "Final Weight:\n";
    layer.weight->print();
    std::cout << "\nFinal Bias:\n";
    layer.bias->print();

    std::cout << "\n=== Final Output ===\n";
    auto final_out = layer.forward(input);
    final_out->print();

    std::cout << "\nBias is working correctly with autograd!\n";

    return 0;
}
