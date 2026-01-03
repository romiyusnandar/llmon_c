#include <iostream>
#include "../include/tensor.h"
#include "../include/ops.h"
#include "../include/optimizer.h"
#include "../include/nn.h"

int main() {
    std::cout << "=== Micro-LLM C++ ===\n";

    auto input = Tensor::create(2, 4);
    input->data = {1.0f, 0.0f, 0.0f, 0.0f,
                   0.0f, 1.0f, 0.0f, 0.0f};

    auto target = Tensor::create(2, 4);
    target->data = {0.0f, 1.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f, 0.0f};

    SelfAttention attention_head(4, 4);
    SGD optimizer(attention_head.parameters(), 0.05f);

    std::cout << "Training Attention Mechanism...\n";

    for (int epoch = 0; epoch <= 200; epoch++) {
        // Forward
        TensorPtr out = attention_head.forward(input);

        // Loss
        TensorPtr loss = mse_loss(out, target);

        // Update
        optimizer.zero_grad();
        loss->backward();
        optimizer.step();

        if (epoch % 50 == 0) {
            std::cout << "Epoch " << epoch << " Loss: " << loss->data[0] << "\n";
        }
    }

    std::cout << "\n=== Final Result ===\n";

    TensorPtr final_out = attention_head.forward(input);
    std::cout << "Input Row 0 (Awalnya [1, 0..]):\n";
    std::cout << "Target Row 0 (Harusnya [0, 1..]):\n";
    final_out->print();

    TensorPtr Q = attention_head.Wq.forward(input);
    TensorPtr K = attention_head.Wk.forward(input);
    TensorPtr Attn = softmax(matmul(Q, transpose(K)));

    std::cout << "Attention Map:\n";
    Attn->print();

    return 0;
}