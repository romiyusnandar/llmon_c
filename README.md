# LLM on C++

A simple yet efficient Large Language Model implementation from scratch in C++ with automatic differentiation (autograd).

## Features

- **Custom Tensor Library** with automatic differentiation
- **Neural Network Layers**: Linear, Embedding, Positional Embedding, Self-Attention
- **Complete GPT Architecture** for text generation
- **Optimizers**: SGD & Adam (with momentum)
- **Operations**: MatMul, ReLU, Softmax, Tanh, Sigmoid
- **Loss Functions**: MSE, Cross-Entropy
- **Scaled Dot-Product Attention** for stable training

## Quick Start

### Compile Main Program
```bash
make
```

### Run Main Program (Self-Attention Demo)
```bash
./main
```

### Run Examples

**Build all examples:**
```bash
make examples
```

**Build & run individual examples:**
```bash
# Build
make gpt_interactive
make gpt_demo
make adam_demo
make test_bias

# Run (after building)
./gpt_interactive
./gpt_demo
./adam_demo
./test_bias
```

### Clean Build
```bash
make clean
```

## 🎯 What Makes This Special?

This is a **fully functional GPT implementation from scratch** with:

### ✅ Complete Architecture
- **Token Embedding** - Converts tokens to vectors
- **Positional Embedding** - Adds position information
- **Self-Attention** - Models relationships between tokens
- **Transformer Block** - Full encoder with FFN + ReLU
- **Output Head** - Projects to vocabulary for predictions

### ✅ Full Autograd System
- Automatic differentiation for all operations
- Backward pass computed automatically
- Gradient accumulation working correctly

### ✅ Training Results
```
Pattern: 1 → 2 → 3 → 1 → 2 → 3 ...

After 500 epochs:
✓ Loss: 0.00001 (converged!)
✓ Predictions: 99.99% accurate
✓ Edge cases handled correctly

Example:
Input:  [1, 2, 3] → Output: [2, 3, 1] ✅
Input:  [2, 3, 1] → Output: [3, 1, 2] ✅
Input:  [3, 1, 2] → Output: [1, 2, 3] ✅
```

### 🚀 Optimizations Applied
1. **Scaled Attention** - `1/√d_k` for stability
2. **Adam Optimizer** - Adaptive learning rates
3. **Efficient MatMul** - Optimized backward pass
4. **Memory Safety** - No dangling pointers
5. **Full Bias Support** - Proper gradient computation

## 📚 Learning Path

1. **Start with**: `make run_adam` - Learn about optimizers
2. **Then try**: `make run_test_bias` - Understand bias training
3. **Finally**: `make run_gpt_interactive` - See full GPT in action!

## License 📄

MIT License - Feel free to use and modify!