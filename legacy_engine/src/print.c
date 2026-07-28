/*
 * print.c
 *
 *  Created on: Sep 22, 2018
 *      Refactored into a leveled, category-based logging subsystem.
 *
 *  The print subsystem offloads all log/hex-dump requests to a
 *  dedicated thread so that:
 *    1. stdout is never interleaved between worker threads
 *    2. the hot search loop is not slowed by I/O syscalls
 */
#include "utility.h"
#include "comm.h"
#include "engine.h"
#include "junqi.h"

#define PRINT_MSG  0
#define MEMOUT_MSG 1

typedef struct PrintMsg
{
	u8 type;
	u8 data[1];
}PrintMsg;

/* Runtime log controls. They can be changed at any moment
 * (e.g. via a COMM_* command or a gdb watchpoint). */
int          g_log_level    = LOG_LEVEL_INFO;
unsigned int g_log_category = LOG_CAT_ALL;

Junqi* gJunqi;

void memout(u8 *pdata,int len)
{
	int i;
	for(i=0;i<len;i++)
	{
		printf("%02X ",*(pdata+i));
		if((i+1)%8==0)
		{
			printf("\n");
		}
	}
	printf("\n");
}

static const char *level_tag(int level)
{
	switch(level)
	{
	case LOG_LEVEL_ERROR: return "ERR";
	case LOG_LEVEL_WARN:  return "WRN";
	case LOG_LEVEL_INFO:  return "INF";
	case LOG_LEVEL_DEBUG: return "DBG";
	case LOG_LEVEL_TRACE: return "TRC";
	default:              return "???";
	}
}

void *print_thread(void *arg)
{
	int len;
	u8 aBuf[REC_LEN];
	Junqi* pJunqi = (Junqi*)arg;
	PrintMsg *pData;

	while (1)
	{
		len = msg_queue_receive(pJunqi->print_qid, (char *)aBuf, REC_LEN);
		if ( len > 0)
		{
			pData = (PrintMsg *)aBuf;
			switch(pData->type)
			{
			case PRINT_MSG:
				aBuf[len] = '\0';
				printf("%s",pData->data);
				break;
			case MEMOUT_MSG:
				memout(pData->data,len-1);
				break;
			default:
				break;
			}
		}
	}

	pthread_detach(pthread_self());
	return NULL;
}

/* Unified, leveled, category-filtered logging entry point.
 * Runtime-filters both by level and by category bitmask. */
void LogMessage(int level, unsigned int category, const char *fmt, ...)
{
	va_list ap;
	char zBuf[256];
	int len;
	Junqi* pJunqi = gJunqi;
	PrintMsg *pData;

	if (level > g_log_level) return;
	if ((category & g_log_category) == 0) return;
	if (pJunqi == NULL || pJunqi->print_qid == NULL) {
		/* Fallback: direct stdout if the print thread is not up yet. */
		va_start(ap, fmt);
		vprintf(fmt, ap);
		printf("\n");
		va_end(ap);
		return;
	}

	va_start(ap, fmt);
	len = vsnprintf(zBuf, sizeof(zBuf) - 1, fmt, ap);
	va_end(ap);
	if (len < 0) return;
	if (len >= (int)sizeof(zBuf) - 1) len = sizeof(zBuf) - 2;
	/* Ensure trailing newline for consistency with legacy log_a. */
	if (len == 0 || zBuf[len - 1] != '\n') {
		zBuf[len++] = '\n';
	}
	zBuf[len] = '\0';

	pData = (PrintMsg *)malloc(len + 1 + sizeof(PrintMsg));
	if (!pData) return;
	pData->type = PRINT_MSG;
	memcpy(pData->data, zBuf, len);
	pData->data[len] = '\0';
	msg_queue_send(pJunqi->print_qid, (char*)pData, len + 1);
	free(pData);

	(void)level_tag;   /* reserved for future prefix formatting */
}

/* Preserve the legacy entry points so existing call sites keep working
 * until they are gradually migrated to LOG_* macros. */
void SafePrint(const char *zFormat, ...)
{
	va_list ap;
	char zBuf[256];
	int len;
	Junqi* pJunqi = gJunqi;
	PrintMsg *pData;

	if (pJunqi == NULL || pJunqi->print_qid == NULL) {
		va_start(ap, zFormat);
		vprintf(zFormat, ap);
		va_end(ap);
		return;
	}

	va_start(ap, zFormat);
	len = vsnprintf(zBuf, sizeof(zBuf), zFormat, ap);
	va_end(ap);
	if (len < 0) return;
	if (len > (int)sizeof(zBuf) - 1) len = sizeof(zBuf) - 1;

	pData = (PrintMsg *)malloc(len + 1 + sizeof(PrintMsg));
	if (!pData) return;
	pData->type = PRINT_MSG;
	memcpy(pData->data, zBuf, len);
	msg_queue_send(pJunqi->print_qid, (char*)pData, len + 1);
	free(pData);
}

void SafeMemout(u8 *aBuf,int len)
{
	PrintMsg *pData;
	Junqi* pJunqi = gJunqi;

	/* Memory dumps are verbose; gate them behind DEBUG level. */
	if (g_log_level < LOG_LEVEL_DEBUG) return;

	if (pJunqi == NULL || pJunqi->print_qid == NULL) {
		memout(aBuf, len);
		return;
	}

	pData = (PrintMsg *)malloc(len + 1 + sizeof(PrintMsg));
	if (!pData) return;
	pData->type = MEMOUT_MSG;
	memcpy(pData->data, aBuf, len);
	msg_queue_send(pJunqi->print_qid, (char*)pData, len + 1);
	free(pData);
}

pthread_t CreatePrintThread(Junqi* pJunqi)
{
	pthread_t tidp = 0;

	pJunqi->print_qid = msg_queue_create();
	if (pJunqi->print_qid == NULL)
	{
		/* Previously: exit(EXIT_FAILURE). We fall back to direct
		 * stdout (LogMessage handles NULL print_qid) rather than
		 * taking down the whole RL environment. */
		fprintf(stderr, "[print] msg_queue_create failed, falling back to sync stdout\n");
		return 0;
	}

	if (pthread_create(&tidp, NULL, (void*)print_thread, pJunqi) != 0) {
		fprintf(stderr, "[print] pthread_create failed\n");
		msg_queue_destroy(pJunqi->print_qid);
		pJunqi->print_qid = NULL;
		return 0;
	}
	return tidp;
}
