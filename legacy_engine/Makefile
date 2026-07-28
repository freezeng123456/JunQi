CC      = gcc
CFLAGS  = -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -g -I./src
LDFLAGS = -lpthread

# Build profile. Override with: make PROFILE=release
PROFILE ?= debug

ifeq ($(PROFILE),release)
	CFLAGS += -O2 -DNDEBUG -DLOG_LEVEL_MAX=LOG_LEVEL_WARN
else ifeq ($(PROFILE),profile)
	CFLAGS += -O2 -pg -DLOG_LEVEL_MAX=LOG_LEVEL_INFO
	LDFLAGS += -pg
else
	# debug (default)
	CFLAGS += -O0 -DLOG_LEVEL_MAX=LOG_LEVEL_TRACE
endif

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))
TARGET = $(BIN_DIR)/JunQiEngine

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)
	@echo "Built $(TARGET) [profile=$(PROFILE)]"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

release:
	$(MAKE) PROFILE=release

debug:
	$(MAKE) PROFILE=debug

.PHONY: all clean release debug
