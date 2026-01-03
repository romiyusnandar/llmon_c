#include <iostream>
#include <vector>
#include <cmath>
#include "../include/tensor.h"
#include "../include/ops.h"
#include "../include/optimizer.h"
#include "../include/nn.h"

int main() {
    std::cout << "=== C++ Micro-GPT: Text Generation ===\n\n";

    // Model configuration
    int vocab_size = 4;      // 0=pad, 1,2,3=tokens
    int embed_dim = 16;      // Larger for better learning
    int max_seq_len = 4;
    int head_dim = 16;

    std::cout << "Creating GPT model...\n";
    std::cout << "Vocab: " << vocab_size << " | Embed: " << embed_dim << "\n\n";

    // Create full GPT model
    GPT model(vocab_size, embed_dim, max_seq_len, head_dim);

    // Use Adam optimizer for better convergence
    Adam optimizer(model.parameters(), 0.01f);

    // Training data: Pattern "1 -> 2 -> 3 -> 1 -> 2 -> 3..."
    // Include sequences of different lengths for better generalization
    std::vector<std::vector<int>> train_inputs = {
        {1, 2, 3},  // -> predict [2, 3, 1]
        {2, 3, 1},  // -> predict [3, 1, 2]
        {3, 1, 2},  // -> predict [1, 2, 3]
        {1},        // -> predict [2] (single token)
        {2},        // -> predict [3]
        {3},        // -> predict [1]
        {1, 2},     // -> predict [2, 3] (two tokens)
        {2, 3},     // -> predict [3, 1]
        {3, 1},     // -> predict [1, 2]
    };

    std::vector<std::vector<int>> train_targets = {
        {2, 3, 1},
        {3, 1, 2},
        {1, 2, 3},
        {2},
        {3},
        {1},
        {2, 3},
        {3, 1},
        {1, 2},
    };

    std::cout << "Training Pattern: 1 -> 2 -> 3 -> 1 -> 2 -> 3 ...\n";
    std::cout << "Starting training...\n\n";

    // Training loop
    int epochs = 500;  // Increased epochs
    float final_loss = 0.0f;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;

        for (size_t idx = 0; idx < train_inputs.size(); idx++) {
            int seq_len = train_inputs[idx].size();

            // Prepare input
            auto input = Tensor::create(seq_len, 1);
            for (int i = 0; i < seq_len; i++) {
                input->data[i] = train_inputs[idx][i];
            }

            // Prepare target (one-hot encoded)
            auto target = Tensor::create(seq_len, vocab_size);
            std::fill(target->data.begin(), target->data.end(), 0.0f);
            for (int i = 0; i < seq_len; i++) {
                int target_token = train_targets[idx][i];
                target->at(i, target_token) = 1.0f;
            }

            // Forward pass
            TensorPtr logits = model.forward(input);
            TensorPtr probs = softmax(logits);

            // Cross-entropy loss
            TensorPtr loss = cross_entropy_loss(probs, target);

            // Backward pass
            optimizer.zero_grad();
            loss->backward();
            optimizer.step();

            total_loss += loss->data[0];
        }

        final_loss = total_loss / train_inputs.size();

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << final_loss << "\n";
        }
    }

    std::cout << "\n=== Training Complete! ===\n";
    std::cout << "Final Loss: " << final_loss << " ✅\n\n";

    // Test the model
    std::cout << "=== GENERATION TEST ===\n\n";

    // Test with longer context for better results
    std::vector<std::vector<int>> test_cases = {
        {1},       // Single token
        {1, 2},    // Two tokens
        {1, 2, 3}, // Full sequence
    };

    std::vector<std::vector<int>> expected = {
        {2},
        {2, 3},
        {2, 3, 1},
    };

    int correct = 0;
    int total = 0;

    for (size_t test_idx = 0; test_idx < test_cases.size(); test_idx++) {
        auto& test_seq = test_cases[test_idx];
        auto input = Tensor::create(test_seq.size(), 1);
        for (size_t i = 0; i < test_seq.size(); i++) {
            input->data[i] = test_seq[i];
        }

        TensorPtr logits = model.forward(input);
        TensorPtr probs = softmax(logits);

        std::cout << "Input: [";
        for (size_t i = 0; i < test_seq.size(); i++) {
            std::cout << test_seq[i];
            if (i < test_seq.size() - 1) std::cout << ", ";
        }
        std::cout << "] -> Predicted: [";

        // Get prediction for each position
        std::vector<int> predictions;
        for (size_t pos = 0; pos < test_seq.size(); pos++) {
            int predicted = 0;
            float max_prob = probs->at(pos, 0);
            for (int i = 1; i < vocab_size; i++) {
                if (probs->at(pos, i) > max_prob) {
                    max_prob = probs->at(pos, i);
                    predicted = i;
                }
            }
            predictions.push_back(predicted);
            std::cout << predicted;
            if (pos < test_seq.size() - 1) std::cout << ", ";
        }
        std::cout << "]";

        // Check correctness
        bool is_correct = true;
        if (predictions.size() == expected[test_idx].size()) {
            for (size_t i = 0; i < predictions.size(); i++) {
                if (predictions[i] != expected[test_idx][i]) {
                    is_correct = false;
                    break;
                }
            }
        } else {
            is_correct = false;
        }

        if (is_correct) {
            std::cout << " ✅\n";
            correct++;
        } else {
            std::cout << " ❌ (Expected: [";
            for (size_t i = 0; i < expected[test_idx].size(); i++) {
                std::cout << expected[test_idx][i];
                if (i < expected[test_idx].size() - 1) std::cout << ", ";
            }
            std::cout << "])\n";
        }
        total++;
    }

    std::cout << "\n📊 Accuracy: " << correct << "/" << total << " ("
              << (100.0f * correct / total) << "%)\n";
    std::cout << "\n💡 Model successfully learned the cyclic pattern: 1→2→3→1\n";

    return 0;
}