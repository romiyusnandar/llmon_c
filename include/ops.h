/**
 * This file contain definition of mathematical operations
 * such as MatMul and ReLU
 */

#ifndef OPS_H
#define OPS_H

#include "tensor.h"

// Forward Pass Operations
Tensor matmul(Tensor& A, Tensor& B);
Tensor relu(Tensor& input); 

#endif