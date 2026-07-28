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
else ifeq ($(PROFILE),sanitize)
	CFLAGS += -O1 -fno-omit-frame-pointer -fsanitize=address,undefined \
		-DLOG_LEVEL_MAX=LOG_LEVEL_WARN
	LDFLAGS += -fsanitize=address,undefined
else
	# debug (default)
	CFLAGS += -O0 -DLOG_LEVEL_MAX=LOG_LEVEL_TRACE
endif

SRC_DIR = src
OBJ_DIR = obj/$(PROFILE)
BIN_DIR = bin

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))
TARGET = $(BIN_DIR)/JunQiEngine
PROFILE_TARGET = $(BIN_DIR)/$(PROFILE)/JunQiEngine

all: $(PROFILE_TARGET)
	@mkdir -p $(BIN_DIR)
	cp $(PROFILE_TARGET) $(TARGET)
	@echo "Built $(TARGET) [profile=$(PROFILE)]"

$(PROFILE_TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)/$(PROFILE)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf obj $(BIN_DIR)

release:
	$(MAKE) PROFILE=release

debug:
	$(MAKE) PROFILE=debug

sanitize:
	$(MAKE) PROFILE=sanitize

.PHONY: all clean release debug sanitize
