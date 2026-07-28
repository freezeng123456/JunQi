/*
 * junqi.h
 *
 *  Created on: Aug 17, 2018
 *      Author: Administrator
 */

#ifndef JUNQI_H_
#define JUNQI_H_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "type.h"
#include <assert.h>
#include "engine.h"
#include "utility.h"
#include "movegen.h"
#include "msg_queue.h"

enum ChessColor {ORANGE,PURPLE,GREEN,BLUE};
enum ChessType {NONE,DARK,JUNQI,DILEI,ZHADAN,SILING,JUNZH,SHIZH,
	            LVZH,TUANZH,YINGZH,LIANZH,PAIZH,GONGB};
enum ChessDir {HOME,RIGHT,OPPS,LEFT};
enum SpcRail {RAIL1=1,RAIL2,RAIL3,RAIL4};
enum RailType {GONGB_RAIL,HORIZONTAL_RAIL,VERTICAL_RAIL,CURVE_RAIL};
enum CompareType {MOVE=1,EAT,BOMB,KILLED,SELECT,SHOW_FLAG,DEAD,BEGIN,TIMER};


#define PLAY_EVENT 0xF5
#define JUMP_EVENT 0x00
#define SURRENDER_EVENT 0x01

#define MOVE_OFFSET (8+30*4)//4字节起始标志+4字节总步数+4家布阵

typedef struct BoardChess BoardChess;
typedef struct ChessLineup
{
	enum ChessDir iDir;//表示棋子是哪家的棋
	//要注意子力越大，type值越小，见ChessType定义
	enum ChessType type;//如果是地方的棋，则表示最小可能
	enum ChessType mx_type;//预测敌方棋的最大可能
	BoardChess *pChess;
	u8 bDead;
	u8 bBomb;
	u8 index;
	u8 isNotLand;
	u8 isNotBomb;
}ChessLineup;

typedef struct BoardPoint
{
	int x;
	int y;
}BoardPoint;


struct BoardChess
{
	enum ChessType type;
	ChessLineup *pLineup;
	int pathCnt;
	u8  pathFlag;
	u8  sameFlag;
	////下面为固定属性，不能改变///////////
	enum SpcRail eCurveRail;
	enum ChessDir iDir;
	int index;
	BoardPoint point;
	u8  isStronghold;
	u8  isCamp;
	u8  isRailway;
	u8  isNineGrid;
};

//邻接表adjacency list;
typedef struct VertexNode AdjNode;
struct VertexNode
{
	BoardChess *pChess;
	AdjNode *pNext;
};

typedef struct BoardGraph
{
	AdjNode *pAdjList;
	int passCnt;
	u8 cnt[28];
}BoardGraph;

typedef struct GraphPath GraphPath;
struct GraphPath
{
	BoardChess *pChess;
	GraphPath *pNext;
	GraphPath *pPrev;
	u8 isHead;
};

typedef struct PartyInfo
{
	u8 bDead;
	u8 cntJump;
	u8 bShowFlag;
	u8 aTypeNum[14];
	u8 aLiveTypeSum[14];//大于某个级别的明棋总和
	u8 aLiveAllNum[14];//大于某个级别的明棋和暗棋总和
}PartyInfo;

/* Network configuration for the engine's UDP socket.
 * All-zero fields mean "use compile-time defaults". Any subset
 * can be overridden via CLI flags (--local-port / --remote-port /
 * --remote-ip) or at runtime by code. This enables running many
 * engine instances on one host for RL self-play. */
typedef struct NetConfig
{
	u16  local_port;    /* UDP port to bind (0 = default)     */
	u16  remote_port;   /* UDP port to send to (0 = default)  */
	char remote_ip[32]; /* Remote host (empty = 127.0.0.1)    */
}NetConfig;

struct Junqi
{
	u8 bStart;
	u8 bStop;
	u8 bGo;
	u8 bSearch;
	u8 bMove;
	u8 iEngineDir;   /* Runtime-configurable engine seat (0..3).
	                  * Replaces compile-time ENGINE_DIR to allow one
	                  * binary to play any seat (needed for RL self-play). */
	enum ChessDir eTurn;
	ChessLineup Lineup[4][30];
	BoardChess ChessPos[4][30];
	BoardChess NineGrid[9];
	//棋盘是17*17，9宫格是5*5
	BoardGraph aBoard[17][17];

	PartyInfo aInfo[4];
	Engine *pEngine;
	MoveList *pMoveList;

	int nRpStep;
	int iRpOfst;
	int begin_time;
	int test_time[2];
	int test_gen_num;
	int test_num;
	int searche_num[2];
	int iKey;
	int test_flag;
	MoveHash **paHash;


	struct sockaddr_in addr;
	int socket_fd;

	/* Per-instance state (moved out of file-scope globals so that
	 * multiple Junqi instances can coexist in one process, e.g. an
	 * embedded Python RL self-play loop). */
	NetConfig netCfg;
	u8  isBoardInit;     /* 1 after the first COMM_INIT bootstrap  */
	int preTurn;         /* previous turn seat (was ::preTurn)     */
	u8  aEventBit[100];  /* event flags bitmap (was ::aEventBit)   */

	MsgQueue* qid;
	MsgQueue* print_qid;
	pthread_mutex_t mutex;
};

Junqi *JunqiOpen(void);
void InitChess(Junqi* pJunqi, u8 *data);
void DestroyAllChess(Junqi *pJunqi, int iDir);
void IncJumpCnt(Junqi *pJunqi, int iDir);
void ChessTurn(Junqi *pJunqi);
void PlayResult(
		Junqi *pJunqi,
		BoardChess *pSrc,
		BoardChess *pDst,
		MoveResultData* pResult
		);
void InitBoard(Junqi* pJunqi);
void InitLineup(Junqi* pJunqi, u8 *data, u8 isInit);
int CheckIfDead(Junqi *pJunqi, int iDir);

#endif /* JUNQI_H_ */
