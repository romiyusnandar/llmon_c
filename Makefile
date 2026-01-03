# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Iinclude

# Folder settings
SRC_DIR = src
OBJ_DIR = build
EXAMPLES_DIR = examples

# Files
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))
TARGET = main

# Library objects (exclude main.o for examples)
LIB_OBJS = $(filter-out $(OBJ_DIR)/main.o, $(OBJS))

# Rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
ifeq ($(OS),Windows_NT)
	@if not exist $(OBJ_DIR) mkdir $(OBJ_DIR)
else
	mkdir -p $(OBJ_DIR)
endif

clean:
ifeq ($(OS),Windows_NT)
	if exist $(OBJ_DIR) rmdir /s /q $(OBJ_DIR)
	if exist $(TARGET).exe del /q $(TARGET).exe
	if exist $(TARGET) del /q $(TARGET)
	if exist adam_demo.exe del /q adam_demo.exe
	if exist adam_demo del /q adam_demo
	if exist test_bias.exe del /q test_bias.exe
	if exist test_bias del /q test_bias
else
	rm -rf $(OBJ_DIR) $(TARGET) adam_demo test_bias
endif

# Build adam_demo example
adam_demo: $(LIB_OBJS) $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -o adam_demo $(EXAMPLES_DIR)/adam_demo.cpp $(LIB_OBJS)

# Run adam_demo
run_adam: adam_demo
ifeq ($(OS),Windows_NT)
	.\adam_demo
else
	./adam_demo
endif

# Build test_bias example
test_bias: $(LIB_OBJS) $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -o test_bias $(EXAMPLES_DIR)/test_bias.cpp $(LIB_OBJS)

# Run test_bias
run_test_bias: test_bias
ifeq ($(OS),Windows_NT)
	.\test_bias
else
	./test_bias
endif

.PHONY: all clean adam_demo run_adam test_bias run_test_bias