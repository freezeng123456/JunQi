/*
 * utility.h
 *
 *  Created on: Sep 22, 2018
 *      Refactored for leveled logging system.
 */

#ifndef UTILITY_H_
#define UTILITY_H_
#include "type.h"
#include "junqi_platform.h"
#include <stdarg.h>
#include <pthread.h>

/* ============================================================
 *  Log Levels (higher value = more verbose)
 * ============================================================ */
#define LOG_LEVEL_NONE   0
#define LOG_LEVEL_ERROR  1
#define LOG_LEVEL_WARN   2
#define LOG_LEVEL_INFO   3
#define LOG_LEVEL_DEBUG  4
#define LOG_LEVEL_TRACE  5

/* Compile-time maximum log level.
 * Set to LOG_LEVEL_NONE to completely disable logging (zero runtime overhead).
 * Set to LOG_LEVEL_TRACE for full verbosity (useful during development). */
#ifndef LOG_LEVEL_MAX
#define LOG_LEVEL_MAX    LOG_LEVEL_INFO
#endif

/* Runtime log level (can be changed on the fly). Declared in print.c */
extern int g_log_level;

/* Category bitmask lets you turn on/off specific subsystems without
 * changing the global verbosity. Declared in print.c */
#define LOG_CAT_CORE      (1 << 0)  /* general flow             */
#define LOG_CAT_SEARCH    (1 << 1)  /* alpha-beta details       */
#define LOG_CAT_COMM      (1 << 2)  /* network / msg queue      */
#define LOG_CAT_RULE      (1 << 3)  /* move legality / compare  */
#define LOG_CAT_EVAL      (1 << 4)  /* evaluation function      */
#define LOG_CAT_PROB      (1 << 5)  /* dark chess probabilities */
#define LOG_CAT_ALL       0xFFFFFFFF

extern unsigned int g_log_category;

/* Core logging API (thread-safe, queued to print thread). */
void LogMessage(int level, unsigned int category, const char *fmt, ...);

#define LOG_ERROR(cat, fmt, ...) \
    do { if (LOG_LEVEL_MAX >= LOG_LEVEL_ERROR) \
        LogMessage(LOG_LEVEL_ERROR, (cat), "[E] " fmt, ## __VA_ARGS__); } while(0)

#define LOG_WARN(cat, fmt, ...) \
    do { if (LOG_LEVEL_MAX >= LOG_LEVEL_WARN) \
        LogMessage(LOG_LEVEL_WARN, (cat), "[W] " fmt, ## __VA_ARGS__); } while(0)

#define LOG_INFO(cat, fmt, ...) \
    do { if (LOG_LEVEL_MAX >= LOG_LEVEL_INFO) \
        LogMessage(LOG_LEVEL_INFO, (cat), "[I] " fmt, ## __VA_ARGS__); } while(0)

#define LOG_DEBUG(cat, fmt, ...) \
    do { if (LOG_LEVEL_MAX >= LOG_LEVEL_DEBUG) \
        LogMessage(LOG_LEVEL_DEBUG, (cat), "[D] " fmt, ## __VA_ARGS__); } while(0)

#define LOG_TRACE(cat, fmt, ...) \
    do { if (LOG_LEVEL_MAX >= LOG_LEVEL_TRACE) \
        LogMessage(LOG_LEVEL_TRACE, (cat), "[T] " fmt, ## __VA_ARGS__); } while(0)

/* ============================================================
 *  Legacy compatibility macros.
 *  log_a -> INFO (core), log_b -> TRACE (prob),
 *  log_c -> direct stdout (blocking, kept for hot-path debug).
 * ============================================================ */
#define log_a(fmt, ...)    LOG_INFO(LOG_CAT_CORE,   fmt, ## __VA_ARGS__)
#define log_b(fmt, ...)    LOG_TRACE(LOG_CAT_PROB,  fmt, ## __VA_ARGS__)
#define log_c(fmt, ...)    printf(fmt "\n", ## __VA_ARGS__)
#define log_fun(fmt, ...)  /* disabled */

/* ============================================================
 *  Binary / hex dump helpers.
 * ============================================================ */
void memout(u8 *pdata, int len);

/* Thread-safe variants. SafePrint/SafeMemout route through the
 * print thread's message queue, avoiding interleaved output. */
void SafePrint(const char *zFormat, ...);
void SafeMemout(u8 *aBuf, int len);

pthread_t CreatePrintThread(Junqi* pJunqi);

/* Backwards-compatible alias for the old typo-ridden name. */
#define CreatPrintThread  CreatePrintThread

#endif /* UTILITY_H_ */
