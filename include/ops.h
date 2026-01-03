/**
 * This file contain definition of mathematical operations
 * such as MatMul and ReLU
 */

#ifndef OPS_H
#define OPS_H

#include "tensor.h"

// Basic Operations
TensorPtr matmul(TensorPtr A, TensorPtr B);
TensorPtr relu(TensorPtr input);
TensorPtr sub(TensorPtr A, TensorPtr B);
TensorPtr transpose(TensorPtr A);
TensorPtr softmax(TensorPtr input);

// Loss Functions
TensorPtr mse_loss(TensorPtr pred, TensorPtr target);
TensorPtr cross_entropy_loss(TensorPtr pred, TensorPtr target);

// Additional Useful Operations
TensorPtr add(TensorPtr A, TensorPtr B);
TensorPtr multiply(TensorPtr A, TensorPtr B); // Element-wise
TensorPtr tanh_activation(TensorPtr input);
TensorPtr sigmoid(TensorPtr input);

#endif