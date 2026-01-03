#ifndef NN_H
#define NN_H

#include "tensor.h"
#include "ops.h"
#include <vector>

/**
 * We use PyTorch concept but in c xd
 * Create base class for all Neural Network layer
 */
class Module {
public:
    virtual ~Module() {}
    virtual TensorPtr forward(TensorPtr input) = 0;
    virtual std::vector<TensorPtr> parameters() = 0;
};

/**
 * First layer, linear fully connected
 * Formula: y = x @ w + b
 */
class Linear : public Module {
public:
    TensorPtr weight;
    TensorPtr bias;
    bool use_bias;

    Linear(int in_features, int out_features, bool bias = true);

    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override;
};

/**
 * Second layer, embedding for text
 * Convert token id (int) to vector
 */
class Embedding : public Module {
public:
    TensorPtr weight;

    Embedding(int num_embeddings, int embedding_dim);

    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override;
};

/**
 * We make simple SelfAttention feature
 */
class SelfAttention : public Module {
public:
    Linear Wq; // For Query projection layer
    Linear Wk; // For Key projection layer
    Linear Wv; // For Value projection layer

    SelfAttention(int embed_dim, int head_dim);

    TensorPtr forward(TensorPtr input) override;
    std::vector<TensorPtr> parameters() override;
};

#endif