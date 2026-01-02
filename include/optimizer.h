#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"
#include <vector>

class SGD {
public:
    std::vector<TensorPtr> parameters;
    float learning_rate;

    SGD(std::vector<TensorPtr> params, float lr)
        : parameters(params), learning_rate(lr) {}

    void step() {
        for (auto& p : parameters) {
            for (size_t i = 0; i < p->data.size(); i++) {
                p->data[i] -= learning_rate * p->grad[i];
            }
        }
    }

    void zero_grad() {
        for (auto& p : parameters) {
            p->zero_grad();
        }
    }
};

#endif