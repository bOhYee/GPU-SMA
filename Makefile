# $@ operand to the left of ':'
# $^ operand to the right of ':'
# $< first operand to the right of ':'
INC_DIR = ./inc
SRC_DIR = ./src
OBJ_DIR = ./obj

CC = nvcc
FLAGS = -I$(INC_DIR) --expt-relaxed-constexpr

DEPS_NAMES = smatch.h
DEPS = $(patsubst %, $(INC_DIR)/%, $(DEPS_NAMES))

OBJ_NAMES = main.o smatch.o
OBJS = $(patsubst %, $(OBJ_DIR)/%, $(OBJ_NAMES))

EXEC_NAME = a.out

# Produce executable
compile: $(OBJS)
	$(CC) -o $(EXEC_NAME) $^ $(FLAGS)

# Compile the .cu files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(DEPS)
	$(CC) -c $< -o $@ $(FLAGS)

# Clean obj directory
clean:
	rm -rd $(OBJ_DIR)/*
	rm $(EXEC_NAME)