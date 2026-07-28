CC = gcc
CFLAGS = -Wall -Wextra -Wno-unused-parameter -g -I./src `pkg-config --cflags gtk+-3.0`
LDFLAGS = `pkg-config --libs gtk+-3.0` -lpthread

PROFILE ?= debug

ifeq ($(PROFILE),release)
	CFLAGS += -O2 -DNDEBUG
else ifeq ($(PROFILE),sanitize)
	CFLAGS += -O1 -fno-omit-frame-pointer -fsanitize=address,undefined
	LDFLAGS += -fsanitize=address,undefined
else
	CFLAGS += -O0
endif

SRC_DIR = src
OBJ_DIR = obj/$(PROFILE)
BIN_DIR = bin

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))
TARGET = $(BIN_DIR)/JunQiGUI
PROFILE_TARGET = $(BIN_DIR)/$(PROFILE)/JunQiGUI

all: $(PROFILE_TARGET) install copy_res

$(PROFILE_TARGET): $(OBJS)
	mkdir -p $(BIN_DIR)/$(PROFILE)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

install: $(PROFILE_TARGET)
	cp $(PROFILE_TARGET) $(TARGET)

copy_res:
	mkdir -p $(BIN_DIR)
	rm -rf $(BIN_DIR)/res $(BIN_DIR)/sound
	cp -r res $(BIN_DIR)/
	cp -r sound $(BIN_DIR)/

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf obj $(BIN_DIR)

release:
	$(MAKE) PROFILE=release

sanitize:
	$(MAKE) PROFILE=sanitize

.PHONY: all clean install release sanitize
