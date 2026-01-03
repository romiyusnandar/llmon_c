#include <iostream>
#include "../include/tensor.h"
#include "../include/ops.h"
#include "../include/optimizer.h"
#include "../include/nn.h"

int main() {
    std::cout << "=== Micro-LLM C++ ===\n";

    // DATASET (Token IDs)
    // Batch size 2. Input ID: 0 dan 1.
    auto input_ids = Tensor::create(2, 1);
    input_ids->data = {0.0f, 1.0f}; // Token ID 0 and 1

    auto target = Tensor::create(2, 2);
    target->data = {1.0f, 1.0f, -1.0f, -1.0f};

    // MODEL DEFINITION
    // Vocab size 2, Embedding 2
    Embedding embed_layer(2, 2);

    std::cout << "Initial Embedding Weights (Random):\n";
    embed_layer.weight->print();

    // OPTIMIZER
    SGD optimizer(embed_layer.parameters(), 0.1f);

    // TRAINING LOOP
    std::cout << "\nTraining Embedding Layer...\n";
    for (int i = 0; i < 100; i++) {

        TensorPtr vectors = embed_layer.forward(input_ids);
        TensorPtr loss = mse_loss(vectors, target);
        optimizer.zero_grad();
        loss->backward();
        optimizer.step();

        if (i % 100 == 0) {
            std::cout << "Epoch " << i << " Loss: " << loss->data[0] << "\n";
        }
    }

    std::cout << "\n=== Final Result ===\n";
    std::cout << "Target: Row 0 -> [1, 1], Row 1 -> [-1, -1]\n";
    std::cout << "Model Weights (Learned):\n";
    embed_layer.weight->print();

    return 0;
}