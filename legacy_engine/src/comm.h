/*
 * comm.h
 *
 *  Created on: Aug 16, 2018
 *      Author: Administrator
 */

#ifndef COMM_H_
#define COMM_H_

#include "junqi_platform.h"
#include <pthread.h>
#include "type.h"
#include "utility.h"

#define COMM_OK          0
#define COMM_ERROR       1
#define COMM_GO          2
#define COMM_MOVE        3
#define COMM_EVNET       4
#define COMM_START       5
#define COMM_READY       6
#define COMM_LINEUP      7
#define COMM_INIT        8
#define COMM_STOP        9
#define COMM_REPLAY      10

#define REC_LEN          200
extern const u8 aMagic[4];

/* Compile-time default UDP ports. Overridable at runtime via
 * pJunqi->netCfg (set by CLI --local-port / --remote-port).
 *
 * Note: the historical TEST build flag (defined in engine.h) used
 * to alternate between port 6678 and 5678 so that TEST/RELEASE
 * builds could share one host. We preserve that behaviour via
 * JUNQI_TEST_BUILD (propagated from engine.h or Makefile). */
#if defined(JUNQI_TEST_BUILD) || defined(TEST)
#define DEFAULT_ENGINE_LOCAL_PORT   6678
#else
#define DEFAULT_ENGINE_LOCAL_PORT   5678
#endif
#define DEFAULT_ENGINE_REMOTE_PORT  1234

typedef struct CommHeader
{
	u8 aMagic[4];
	u8 iDir;
	u8 eFun;
	u8 reserve[2];
}CommHeader;

typedef struct MoveResultData
{
	u8 src[2];
	u8 dst[2];
	u8 result;
	u8 extra_info;//0~2bit的含义：0：军旗阵亡 1：src是司令 2：dst是司令
	u8 junqi_src[2];
	u8 junqi_dst[2];
}MoveResultData;


pthread_t CreateCommThread(Junqi* pJunqi);
/* Backwards-compatible typo alias (deprecated, use CreateCommThread). */
#define CreatCommThread  CreateCommThread
void SendHeader(Junqi* pJunqi, u8 iDir, u8 eFun);
void SendMove(Junqi* pJunqi, BoardChess *pSrc, BoardChess *pDst);
void SendEvent(Junqi* pJunqi, int iDir, u8 event);
void SetRecLineup(Junqi* pJunqi, u8 *data, int iDir);

#endif /* COMM_H_ */
