/**
 * Copyright: @romiyusnandar
 * When we using python and type 'x = torch.randn(10, 10)'
 * There's a lot going on in memory, in C++ we have to manage it manually
 *
 * We need a container to save:
 *      1. Data, weights value or activation
 *      2. Grand, gradient value for learning process
 *      3. Shape, matrix dimensions (rowa, cloumns)
 */

#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <memory>
#include <functional>
#include <set>

struct Tensor;

using TensorPtr = std::shared_ptr<Tensor>;

struct Tensor {
    int rows;
    int cols;
    std::vector<float> data;
    std::vector<float> grad;

    std::vector<TensorPtr> prev;

    std::function<void()> _backward;

    Tensor(int r, int c);
    static TensorPtr create(int r, int c);

    // Methods
    void random_init();
    void zero_grad();
    void backward();

    float& at(int i, int j);
    float& grad_at(int i, int j); 
    void print() const;
};

#endif