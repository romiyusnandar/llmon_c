#include <iostream>
#include "../include/tensor.h"
#include "../include/ops.h"
#include "../include/optimizer.h"
#include "../include/nn.h"

int main() {
    std::cout << "=== Micro-LLM C++ ===\n";

    // 1. DATA INPUT (Anggap ini hasil embedding)
    // Sequence Length = 2, Embedding Dim = 4
    auto input = Tensor::create(2, 4);
    // Token A: [1, 0, 0, 0], Token B: [0, 1, 0, 0]
    input->data = {1.0f, 0.0f, 0.0f, 0.0f,
                   0.0f, 1.0f, 0.0f, 0.0f};

    // 2. TARGET
    // Kita ingin Token A (Baris 0) outputnya menjadi seperti Token B
    // Artinya: Token A harus "attend" ke Token B.
    auto target = Tensor::create(2, 4);
    // Target Baris 0 = [0, 1, 0, 0] (Meniru Token B)
    // Target Baris 1 = [0, 1, 0, 0] (Tetap dirinya sendiri)
    target->data = {0.0f, 1.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f, 0.0f};

    // 3. MODEL: Single Head Attention
    // Input 4 dim -> Output 4 dim
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
    // Kita cek Output
    TensorPtr final_out = attention_head.forward(input);
    std::cout << "Input Row 0 (Awalnya [1, 0..]):\n";
    std::cout << "Target Row 0 (Harusnya [0, 1..]):\n";
    final_out->print();

    // Mari kita intip "Attention Matrix" (Scores)
    // Ini manual forward untuk debug
    TensorPtr Q = attention_head.Wq.forward(input);
    TensorPtr K = attention_head.Wk.forward(input);
    TensorPtr Attn = softmax(matmul(Q, transpose(K)));

    std::cout << "Attention Map (Siapa melihat siapa?):\n";
    Attn->print();
    std::cout << "(Baris 0 Kolom 1 harusnya tinggi, artinya Token 0 melihat Token 1)\n";

    return 0;
}