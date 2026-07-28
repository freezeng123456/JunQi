/*
 * engine.h
 *
 *  Created on: Aug 18, 2018
 *      Author: Administrator
 */

#ifndef ENGINE_H_
#define ENGINE_H_
#include "type.h"
#include <unistd.h>
#include <fcntl.h>
#include "evaluate.h"
#include "comm.h"
#include "movegen.h"

enum MoveEvent{
	CAMP_EVENT,
	MOVE_EVENT,
	GONGB_EVENT,
	DARK_EVENT,
	EAT_EVENT,
	JUNQI_EVENT,
	BOMB_EVENT
};

/* ============================================================
 *  Engine seat configuration.
 *
 *  Historically ENGINE_DIR was a compile-time constant that picked
 *  one of the four board seats (0~3) for the AI. That made it
 *  impossible to run self-play (four AIs in one process) and
 *  inconvenient for reinforcement-learning training loops.
 *
 *  The new model:
 *    - Each Junqi* has its own iEngineDir field (set at InitBoard).
 *    - ENGINE_DIR_DEFAULT is the initial seat at boot time, chosen
 *      by the TEST build flag for backwards compatibility.
 *    - All call sites should use JQ_ENGINE_DIR(pJunqi) going forward.
 *
 *  The legacy ENGINE_DIR macro is kept as an alias that falls back
 *  to the global gJunqi so existing code compiles unchanged.
 * ============================================================ */
#define TEST

#ifdef  TEST
#define ENGINE_DIR_DEFAULT   0
#else
#define ENGINE_DIR_DEFAULT   1
#endif

#define JQ_ENGINE_DIR(pJunqi)  ((pJunqi)->iEngineDir)

/* Legacy alias: reads the per-instance seat through the global
 * gJunqi. This keeps the 30+ existing references working while we
 * migrate them incrementally. */
extern Junqi* gJunqi;
#define ENGINE_DIR             (gJunqi ? gJunqi->iEngineDir : ENGINE_DIR_DEFAULT)

#define INFINITY 10000
extern u8 aEventBit[100];

typedef struct MoveResult
{
    MoveResultData move;
    int percent;
    u8 flag;//标记是移动还是碰撞
}MoveResult;

typedef struct  BestMoveList  BestMoveList;
struct  BestMoveList
{
    MoveResult result[4];
    BestMoveList *pNext;
};

struct BestMove
{
    BestMoveList *pHead;
    BestMoveList *pNode;
    MoveList *pTest;
    u8 flag1; //判断是否已经搜索过
    u8 flag2; //move是否不为空
    u8 mxPerFlag;
    u8 mxPerFlag1;
};

typedef struct ENGINE
{
	Junqi *pJunqi;
	//++++++++++++++
	//早期的代码，现在不用
	BoardChess *pCamp[2];
	BoardChess *pBomb[2];
	BoardChess *pEat[2];
	BoardChess *pMove[2];
	GraphPath *pPath[2];//pPath[0] 暂时不用
	u16 eventId;
    u8  eventFlag;
    //--------------------------
    BestMove aBestMove[30];
    BoardChess *pBest[2];
    PositionList *pPos;
    Value_Parameter valPara;
    int searchCnt; // Recursive depth counter for AlphaBeta search
}Engine;

typedef struct EventHandle
{
	u8 (*xEventFun)(Engine *pEngine);
	u16  eventId;
}EventHandle;


pthread_t CreateEngineThread(Junqi* pJunqi);
/* Backwards-compatible typo alias (deprecated, use CreateEngineThread). */
#define CreatEngineThread  CreateEngineThread
void SendEvent(Junqi* pJunqi, int iDir, u8 event);
Engine *OpenEngine(Junqi *pJunqi);
void CloseEngine(Engine *pEngine);

#endif /* ENGINE_H_ */
