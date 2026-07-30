CC      = gcc
CFLAGS  = -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -g -I./src -I../legacy_common
LDFLAGS = -lpthread

# Set WINDOWS=1 for the MSYS2/MinGW build. Windows binaries stay in their
# own directory so they never overwrite the Mac/Linux build output.
WINDOWS ?= 0
ifeq ($(WINDOWS),1)
	LDFLAGS += -lws2_32
	BIN_DIR = bin/windows
	WIN_SUFFIX = .exe
	OBJ_SUFFIX = -windows
else
	BIN_DIR = bin
	WIN_SUFFIX =
	OBJ_SUFFIX =
endif

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
OBJ_DIR = obj/$(PROFILE)$(OBJ_SUFFIX)

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))
TARGET = $(BIN_DIR)/JunQiEngine$(WIN_SUFFIX)
PROFILE_TARGET = $(BIN_DIR)/$(PROFILE)/JunQiEngine$(WIN_SUFFIX)

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
