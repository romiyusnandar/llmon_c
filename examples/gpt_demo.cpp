/**
 * GPT Demo: Text Generation
 * Task: Teach model the pattern "1, 2, 3, 1, 2, 3..."
 * Given "1", model should predict "2"
 * Given "2", model should predict "3"
 * Given "3", model should predict "1"
 */

#include <iostream>
#include <vector>
#include "../include/tensor.h"
#include "../include/nn.h"
#include "../include/optimizer.h"

int main() {
    std::cout << "=== GPT Text Generation Demo ===\n\n";

    // Vocabulary: 0=padding, 1=token_1, 2=token_2, 3=token_3
    int vocab_size = 4;
    int embed_dim = 8;
    int max_seq_len = 4;
    int head_dim = 8;

    // Create GPT model
    std::cout << "Creating GPT model...\n";
    std::cout << "Vocab Size: " << vocab_size << "\n";
    std::cout << "Embedding Dim: " << embed_dim << "\n";
    std::cout << "Max Sequence Length: " << max_seq_len << "\n\n";

    GPT model(vocab_size, embed_dim, max_seq_len, head_dim);

    // Use Adam optimizer for better convergence
    Adam optimizer(model.parameters(), 0.01f);

    // Training data: Pattern "1, 2, 3, 1, 2, 3..."
    // Input sequence: [1, 2, 3] -> Target: [2, 3, 1]
    std::vector<std::vector<int>> train_inputs = {
        {1, 2, 3},  // -> predict [2, 3, 1]
        {2, 3, 1},  // -> predict [3, 1, 2]
        {3, 1, 2},  // -> predict [1, 2, 3]
    };

    std::vector<std::vector<int>> train_targets = {
        {2, 3, 1},
        {3, 1, 2},
        {1, 2, 3},
    };

    std::cout << "Training Data:\n";
    std::cout << "Pattern: 1 -> 2 -> 3 -> 1 -> 2 -> 3 ...\n\n";

    // Training loop
    std::cout << "Starting training...\n";
    int epochs = 500;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;

        // Train on each sequence
        for (size_t idx = 0; idx < train_inputs.size(); idx++) {
            // Prepare input tensor
            auto input = Tensor::create(3, 1);
            input->data[0] = train_inputs[idx][0];
            input->data[1] = train_inputs[idx][1];
            input->data[2] = train_inputs[idx][2];

            // Prepare target tensor (one-hot encoded)
            auto target = Tensor::create(3, vocab_size);
            std::fill(target->data.begin(), target->data.end(), 0.0f);
            for (int i = 0; i < 3; i++) {
                int target_token = train_targets[idx][i];
                target->at(i, target_token) = 1.0f;
            }

            // Forward pass
            TensorPtr logits = model.forward(input);

            // Apply softmax to get probabilities
            TensorPtr probs = softmax(logits);

            // Cross-entropy loss
            TensorPtr loss = cross_entropy_loss(probs, target);

            // Backward pass
            optimizer.zero_grad();
            loss->backward();
            optimizer.step();

            total_loss += loss->data[0];
        }

        // Print progress
        if (epoch % 50 == 0) {
            std::cout << "Epoch " << epoch << " | Avg Loss: " << total_loss / train_inputs.size() << "\n";
        }
    }

    std::cout << "\n=== Training Complete! ===\n\n";

    // Test the model
    std::cout << "=== Testing Model ===\n\n";

    std::vector<std::vector<int>> test_inputs = {
        {1, 2, 3},
        {2, 3, 1},
        {3, 1, 2},
    };

    for (auto& test_seq : test_inputs) {
        auto input = Tensor::create(3, 1);
        input->data[0] = test_seq[0];
        input->data[1] = test_seq[1];
        input->data[2] = test_seq[2];

        TensorPtr logits = model.forward(input);
        TensorPtr probs = softmax(logits);

        std::cout << "Input: [" << test_seq[0] << ", " << test_seq[1] << ", " << test_seq[2] << "]\n";
        std::cout << "Predictions:\n";

        for (int i = 0; i < 3; i++) {
            // Find token with highest probability
            int predicted = 0;
            float max_prob = probs->at(i, 0);
            for (int j = 1; j < vocab_size; j++) {
                if (probs->at(i, j) > max_prob) {
                    max_prob = probs->at(i, j);
                    predicted = j;
                }
            }

            std::cout << "  Position " << i << ": Token " << predicted
                      << " (prob: " << max_prob << ")\n";
        }
        std::cout << "\n";
    }

    std::cout << "💡 If trained well, model should predict:\n";
    std::cout << "   [1,2,3] -> [2,3,1]\n";
    std::cout << "   [2,3,1] -> [3,1,2]\n";
    std::cout << "   [3,1,2] -> [1,2,3]\n";

    return 0;
}
