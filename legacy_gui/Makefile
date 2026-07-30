CC = gcc
CFLAGS = -Wall -Wextra -Wno-unused-parameter -g -I./src -I../legacy_common `pkg-config --cflags gtk+-3.0`
LDFLAGS = `pkg-config --libs gtk+-3.0` -lpthread

# Set WINDOWS=1 for the MSYS2/MinGW build. Windows binaries stay in their
# own directory so they never overwrite the Mac/Linux build output.
WINDOWS ?= 0
ifeq ($(WINDOWS),1)
	LDFLAGS += -lws2_32 -lwinmm
	BIN_DIR = bin/windows
	WIN_SUFFIX = .exe
	OBJ_SUFFIX = -windows
else
	BIN_DIR = bin
	WIN_SUFFIX =
	OBJ_SUFFIX =
endif

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
OBJ_DIR = obj/$(PROFILE)$(OBJ_SUFFIX)

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))
TARGET = $(BIN_DIR)/JunQiGUI$(WIN_SUFFIX)
PROFILE_TARGET = $(BIN_DIR)/$(PROFILE)/JunQiGUI$(WIN_SUFFIX)

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
