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
	if exist gpt_demo.exe del /q gpt_demo.exe
	if exist gpt_demo del /q gpt_demo
else
	rm -rf $(OBJ_DIR) $(TARGET) adam_demo test_bias gpt_demo
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

# Build gpt_demo example
gpt_demo: $(LIB_OBJS) $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -o gpt_demo $(EXAMPLES_DIR)/gpt_demo.cpp $(LIB_OBJS)

# Run gpt_demo
run_gpt: gpt_demo
ifeq ($(OS),Windows_NT)
	.\gpt_demo
else
	./gpt_demo
endif

# Build gpt_interactive example
gpt_interactive: $(LIB_OBJS) $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -o gpt_interactive $(EXAMPLES_DIR)/gpt_interactive.cpp $(LIB_OBJS)

# Run gpt_interactive
run_gpt_interactive: gpt_interactive
ifeq ($(OS),Windows_NT)
	.\gpt_interactive
else
	./gpt_interactive
endif

.PHONY: all clean adam_demo run_adam test_bias run_test_bias gpt_demo run_gpt gpt_interactive run_gpt_interactive