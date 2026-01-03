# Micro-LLM C++ 🚀

A simple yet efficient Large Language Model implementation from scratch in C++ with automatic differentiation (autograd).

## Features ✨

- **Custom Tensor Library** with automatic differentiation
- **Neural Network Layers**: Linear, Embedding, Self-Attention
- **Optimizers**: SGD & Adam (with momentum)
- **Operations**: MatMul, ReLU, Softmax, Tanh, Sigmoid
- **Loss Functions**: MSE, Cross-Entropy
- **Scaled Dot-Product Attention** for stable training

## Quick Start 🏃‍♂️

### Compile Main Program
```bash
make
```

### Run Main Program (Self-Attention Demo)
```bash
./main
```

### Run Examples

**Adam vs SGD Comparison:**
```bash
make run_adam
```

**Linear Layer with Bias Test:**
```bash
make run_test_bias
```

Or compile manually:
```bash
make adam_demo
./adam_demo

make test_bias
./test_bias
```

### Clean Build
```bash
make clean
```

## License 📄

MIT License - Feel free to use and modify!