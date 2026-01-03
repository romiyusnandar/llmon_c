/**
 * This file contain definition of mathematical operations
 * such as MatMul and ReLU
 */

#ifndef OPS_H
#define OPS_H

#include "tensor.h"

// Input Pointer -> Output Pointer
TensorPtr matmul(TensorPtr A, TensorPtr B);
TensorPtr relu(TensorPtr input);

// Phase 3
TensorPtr sub(TensorPtr A, TensorPtr B);
TensorPtr mse_loss(TensorPtr pred, TensorPtr target);

TensorPtr transpose(TensorPtr A);
TensorPtr softmax(TensorPtr input);

#endif