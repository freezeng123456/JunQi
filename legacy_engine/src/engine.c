/*
 * engine.c
 *
 *  Created on: Aug 18, 2018
 *      Author: Administrator
 */
#include "junqi.h"
#include "comm.h"
#include <time.h>
#include "event.h"
#include "engine.h"
#include "path.h"
#include "evaluate.h"
#include "movegen.h"
#include "search.h"

/* preTurn was formerly a file-scope global; now per-instance on Junqi. */

EventHandle eventArr[] = {
	{ ComeInCamp, CAMP_EVENT },
	{ ProBombEvent, BOMB_EVENT },
	{ ProEatEvent, EAT_EVENT },
	{ ProEatEvent, GONGB_EVENT },
	{ ProEatEvent, DARK_EVENT },
	{ ProJunqiEvent, MOVE_EVENT },
	{ ProJunqiEvent, JUNQI_EVENT }
};


/* RNG shared across the process. Seeded once on first use with a
 * mix of time + pid + seat so that multiple engine instances started
 * within the same second still get distinct seeds - critical for RL
 * self-play reproducibility & sample diversity. */
u32 random_(void)
{
	static u32 x = 0;
	static u8 isInit = 0;
	if(!isInit)
	{
		isInit = 1;
		/* Mix several entropy sources to avoid collision when 4
		 * seats are spawned in parallel from a shell script. */
		unsigned int t   = (unsigned int)time(NULL);
		unsigned int pid = (unsigned int)getpid();
		unsigned int seat = gJunqi ? (unsigned int)gJunqi->iEngineDir : 0;
		unsigned int seed = t ^ (pid * 2654435761u) ^ (seat << 16);
		LOG_DEBUG(LOG_CAT_CORE, "random_ seeded with %u (t=%u pid=%u seat=%u)",
		          seed, t, pid, seat);
		srand(seed);
	}
    for(int i=0; i<20; i++)
    {
    	x ^= (rand()&1)<<i;
    }
    x++;
    return x;
}

/* Explicit RNG seeding hook for reproducible RL experiments.
 * Call this before JunqiOpen() if deterministic behaviour is desired. */
void random_seed(unsigned int seed)
{
	srand(seed);
}

void ProMoveEvent(Junqi* pJunqi, u8 iDir, u8 event)
{
	if( event==JUMP_EVENT )
	{
		LOG_DEBUG(LOG_CAT_CORE, "player %d jumps (jump_cnt=%d)",
		          iDir, pJunqi->aInfo[iDir].cntJump + 1);
		assert( iDir==pJunqi->eTurn );
		IncJumpCnt(pJunqi, iDir);
		ChessTurn(pJunqi);
	}
	else if( event==SURRENDER_EVENT )
	{
		LOG_INFO(LOG_CAT_CORE, "player %d surrenders", iDir);
		DestroyAllChess(pJunqi, iDir);
		if( iDir==pJunqi->eTurn )
		{
			ChessTurn(pJunqi);
		}
	}
	else
	{
		LOG_WARN(LOG_CAT_CORE, "unknown move event 0x%02X from dir=%d", event, iDir);
	}
}

Engine *OpenEngine(Junqi *pJunqi)
{
	Engine *pEngine = (Engine *)malloc(sizeof(Engine));
	if (pEngine == NULL) {
		LOG_ERROR(LOG_CAT_CORE, "OpenEngine: malloc failed");
		return NULL;
	}
	memset(pEngine, 0, sizeof(Engine));
	memset(aEventBit, 0, sizeof(aEventBit));
	pEngine->pJunqi = pJunqi;
	InitValuePara(&pEngine->valPara);
	LOG_INFO(LOG_CAT_CORE, "OpenEngine: engine created for seat=%d",
	         JQ_ENGINE_DIR(pJunqi));
	return pEngine;
}

void CloseEngine(Engine *pEngine)
{
	if(pEngine!=NULL)
	{
		LOG_INFO(LOG_CAT_CORE, "CloseEngine: releasing engine");
		pEngine->pJunqi->pEngine = NULL;
		free(pEngine);
	}
}

void CheckMoveEvent(
	Engine *pEngine,
	BoardChess *pSrc,
	BoardChess *pDst,
	MoveResultData* pResult)
{
	int type = pResult->result;
	if( type==MOVE || type==EAT  )
	{
		if( pDst->pLineup->iDir%2!=ENGINE_DIR%2 )
		{
			CheckCampEvent(pEngine,pDst);
		}
	}
	CheckBombEvent(pEngine);
	CheckEatEvent(pEngine);
	CheckJunqiEvent(pEngine);
}

u8 DealEvent(Engine *pEngine)
{
	u8 isMove = 0;
	int i;
	u8 eventFlag = 0;
	u8 eventId = 0;
	u8 index;

	for(i=0; i<sizeof(eventArr)/sizeof(eventArr[0]); i++)
	{
		if( TESTBIT(aEventBit, eventArr[i].eventId) )
		{
			if( eventArr[i].eventId>=eventId )
			{
				eventId = eventArr[i].eventId;
				index = i;
			}
			eventFlag = 1;
		}
	}

	if( eventFlag )
	{
		isMove = eventArr[index].xEventFun(pEngine);
	}

	return isMove;
}


void ProMoveResult(Junqi* pJunqi, u8 iDir, u8 *data)
{
	BoardChess *pSrc, *pDst;
	BoardPoint p1,p2;
	MoveResultData *pResult = (MoveResultData*)data;

	p1.x = pResult->src[0]%17;
	p1.y = pResult->src[1]%17;
	p2.x = pResult->dst[0]%17;
	p2.y = pResult->dst[1]%17;
	if( pJunqi->aBoard[p1.x][p1.y].pAdjList && pJunqi->aBoard[p2.x][p2.y].pAdjList )
	{
		pSrc = pJunqi->aBoard[p1.x][p1.y].pAdjList->pChess;
		pDst = pJunqi->aBoard[p2.x][p2.y].pAdjList->pChess;
		if( pSrc==NULL || pDst==NULL )
		{
			LOG_ERROR(LOG_CAT_COMM, "ProMoveResult: src/dst chess is NULL at (%d,%d)->(%d,%d)",
			          p1.x, p1.y, p2.x, p2.y);
			SendHeader(pJunqi, iDir, COMM_ERROR);
			return;
		}
	}
	else
	{
		LOG_ERROR(LOG_CAT_COMM, "ProMoveResult: empty board vertex at (%d,%d) or (%d,%d)",
		          p1.x, p1.y, p2.x, p2.y);
		SendHeader(pJunqi, iDir, COMM_ERROR);
		return;
	}
	LOG_DEBUG(LOG_CAT_CORE, "move dir=%d (%d,%d)->(%d,%d) result=%d",
	          iDir, p1.x, p1.y, p2.x, p2.y, pResult->result);
	assert( pSrc->pLineup->iDir==iDir );
	PlayResult(pJunqi, pSrc, pDst, pResult);

	if( pJunqi->bStart )
	{
		// Check if any player has no valid moves after this move
		for(int i=0; i<4; i++)
		{
			if( !pJunqi->aInfo[i].bDead && CheckIfDead(pJunqi, i) )
			{
				LOG_INFO(LOG_CAT_CORE, "player %d has no legal moves -> dead", i);
			}
		}
		ChessTurn(pJunqi);
		//CheckMoveEvent(pJunqi->pEngine, pSrc, pDst, pResult);
	}


}

BoardChess * GetMoveDst(Junqi* pJunqi, BoardChess *pSrc)
{
	BoardChess *pDst=NULL;
	BoardChess *pTemp;
	u32 rand = 0;
	int i,j;

	rand = random_()%129;
	//rand = 109;
	for(i=0; i<129; i++)
	{
		//log_a("aa i %d rand %d k %d",i, rand,(rand+i)%129);
		j = (i+rand)%129;
		if( j<120 )
		{
			assert(j/30>=0&&j/30<4);
			assert(j%30>=0&&j%30<30);
			pTemp = &pJunqi->ChessPos[j/30][j%30];
		}
		else
		{
			assert(j-120>=0&&j-120<9);
			pTemp = &pJunqi->NineGrid[j-120];
		}
		if( pTemp->type!=NONE && pSrc->pLineup->iDir%2==pTemp->pLineup->iDir%2 )
		{
			continue;
		}
		if( IsEnableMove(pJunqi, pSrc,pTemp) )
		{
			pDst = pTemp;
			break;
		}
	}

	return pDst;
}

void SendRandMove(Junqi* pJunqi)
{
    u32 rand;
    int i;
    BoardChess *pSrc;
    BoardChess *pDst;
    ChessLineup *pLineup;

    rand = random_()%30;
    for(i=0;  i<30; i++)
    {
    	pLineup = &pJunqi->Lineup[pJunqi->eTurn][(rand+i)%30];
    	if( pLineup->bDead )
    	{
    		continue;
    	}
    	pSrc = pLineup->pChess;
    	LOG_TRACE(LOG_CAT_CORE, "SendRandMove try i=%d rand=%d idx=%d type=%d",
    	          i, rand, (rand+i)%30, pLineup->type);
    	if(pLineup->type!=NONE && pLineup->type!=JUNQI && pLineup->type!=DILEI )
    	{
    		pDst = GetMoveDst(pJunqi, pSrc);
    		if( pDst!=NULL )
    		{
    			LOG_DEBUG(LOG_CAT_CORE, "SendRandMove: (%d,%d)->(%d,%d)",
    			          pSrc->point.x, pSrc->point.y,
    			          pDst->point.x, pDst->point.y);
    			SendMove(pJunqi, pSrc, pDst);
    			return;
    		}
    	}
    }
    LOG_INFO(LOG_CAT_CORE, "SendRandMove: no move available, jump for seat %d",
             pJunqi->eTurn);
    SendEvent(pJunqi, pJunqi->eTurn, JUMP_EVENT);

}

void ClearBestMoveFlag(Engine *pEngine)
{
    for(int i=0; i<30; i++)
    {
        pEngine->aBestMove[i].flag1 = 0;
        pEngine->aBestMove[i].mxPerFlag = 0;
        pEngine->aBestMove[i].mxPerFlag1 = 0;
    }
}

void ProRecMsg(Junqi* pJunqi, u8 *data)
{
	CommHeader *pHead;
	pHead = (CommHeader *)data;
	u8 event;
	u8 isMove = 0;
	int value;
	int eTurn;
	Engine *pEngine = pJunqi->pEngine;
	int i;

	if( memcmp(pHead->aMagic, aMagic, 4)!=0 )
	{
		return;
	}

	switch(pHead->eFun)
	{
	case COMM_EVNET:
		pJunqi->preTurn = pJunqi->eTurn;
		event = *((u8*)&pHead[1]);
		ProMoveEvent(pJunqi, pHead->iDir, event);
		SendHeader(pJunqi, pHead->iDir, COMM_OK);
		break;
	case COMM_MOVE:
		pJunqi->preTurn = pJunqi->eTurn;
		//log_c("turn %d %d",pHead->iDir,pJunqi->eTurn);
		assert( pHead->iDir==pJunqi->eTurn );
		data = (u8*)&pHead[1];

		ProMoveResult(pJunqi, pHead->iDir, data);
		SendHeader(pJunqi, pHead->iDir, COMM_OK);
		break;
	case COMM_REPLAY:
		log_b("reply %d",pHead->iDir);
		SendHeader(pJunqi, pHead->iDir, COMM_REPLAY);
		pJunqi->eTurn = pHead->iDir;
		pJunqi->bStart = 1;
		//在COMM_START指令中清0
		pJunqi->nRpStep = *((u16*)pHead->reserve);
		pJunqi->iRpOfst = 0;

		//获取复盘布阵，以后可能有用
//		data = (u8*)&pHead[1];
//		InitReplyLineup(pJunqi,&data[8]);

		break;
	default:
		break;
	}

	if( !pJunqi->bStart || pJunqi->bStop )
	{
		return;
	}
    if( 0==pJunqi->nRpStep ||
    	pJunqi->iRpOfst>pJunqi->nRpStep-1 )
    {
    	eTurn = pJunqi->eTurn;
    	LOG_INFO(LOG_CAT_SEARCH, "=== search start: turn=%d engine_seat=%d ===",
    	         eTurn, JQ_ENGINE_DIR(pJunqi));
    	pJunqi->bGo = 0;
    	pJunqi->bMove = 0;
    	pJunqi->begin_time = (unsigned int)time(NULL);

    	memset(pEngine->aBestMove,0,sizeof(pEngine->aBestMove));

    	for(i=0; i<5; i++)
    	{
    		int depth_start_time = (unsigned int)time(NULL);
    		pJunqi->eTurn = eTurn;
    		pthread_mutex_lock(&pJunqi->mutex);
    		pJunqi->bSearch = 1;
    		pJunqi->test_num = 0;
    		pJunqi->test_gen_num = 0;
    		pJunqi->searche_num[0] = 0;
    		pJunqi->searche_num[1] = 0;
    		ClearBestMoveFlag(pEngine);

			value = AlphaBeta1(pJunqi,i,-INFINITY,INFINITY);
			pJunqi->bSearch = 0;
			pthread_mutex_unlock(&pJunqi->mutex);

			LOG_DEBUG(LOG_CAT_SEARCH,
			          "depth=%d nodes=%d gens=%d hash_hits=%d/%d elapsed=%ds",
			          i, pJunqi->test_num, pJunqi->test_gen_num,
			          pJunqi->searche_num[0], pJunqi->searche_num[1],
			          (int)(time(NULL) - depth_start_time));

			BoardChess **pBest = pJunqi->pEngine->pBest;
			if(i>0 && pBest[0] && pBest[1])
			{
				LOG_DEBUG(LOG_CAT_SEARCH, "best move: (%d,%d)->(%d,%d)",
				          pBest[0]->point.x, pBest[0]->point.y,
				          pBest[1]->point.x, pBest[1]->point.y);
			}

			if( TimeOut(pJunqi) )
			{
				LOG_INFO(LOG_CAT_SEARCH, "time out at depth=%d", i);
				break;
			}
			if( eTurn%2!=JQ_ENGINE_DIR(pJunqi)%2 )
			{
				value = -value;
			}

			LOG_INFO(LOG_CAT_SEARCH, "depth=%d value=%d", i, value);
    	}

    	FreeBestMoveList(pEngine->aBestMove,i);
    	LOG_INFO(LOG_CAT_SEARCH, "=== search end: total_elapsed=%ds ===",
    	         (int)(time(NULL) - pJunqi->begin_time));

    	pJunqi->eTurn = eTurn;

    }
    pJunqi->bMove = 0;
    pJunqi->iRpOfst++;

    if( pJunqi->preTurn == pJunqi->eTurn )
	{
		return;
	}
	pJunqi->bGo = 0;

	if( pJunqi->eTurn%2==JQ_ENGINE_DIR(pJunqi)%2 )
	{
		if( pJunqi->aInfo[pJunqi->eTurn].bDead )
		{
			ChessTurn(pJunqi);
		}

		//isMove = DealEvent(pJunqi->pEngine);
		isMove = SendBestMove(pJunqi->pEngine);

		if( !isMove )
		{
			SendRandMove(pJunqi);
		}

	}
}

void *engine_thread(void *arg)
{
	int len;
	u8 aBuf[REC_LEN];
	Junqi* pJunqi = (Junqi*)arg;

	LOG_INFO(LOG_CAT_CORE, "engine thread started");
    while (1)
    {
    	len = msg_queue_receive(pJunqi->qid, (char *)aBuf, REC_LEN);
        if ( len > 0)
        {
        	LOG_TRACE(LOG_CAT_COMM, "engine queue recv %d bytes", len);
        	SafeMemout(aBuf,len);
        	ProRecMsg(pJunqi, aBuf);
        }
    }

	pthread_detach(pthread_self());
	return NULL;
}

pthread_t CreateEngineThread(Junqi* pJunqi)
{
    pthread_t tidp = 0;

    pJunqi->qid = msg_queue_create();
    if (pJunqi->qid == NULL)
    {
        /* Previously: exit(EXIT_FAILURE). In RL self-play, the
         * parent orchestrator should detect a non-starting engine
         * via missing COMM_READY and recover, instead of us killing
         * the whole process tree. */
        LOG_ERROR(LOG_CAT_CORE, "CreateEngineThread: msg_queue_create failed");
        return 0;
    }

    if (pthread_create(&tidp, NULL, (void*)engine_thread, pJunqi) != 0) {
        LOG_ERROR(LOG_CAT_CORE, "CreateEngineThread: pthread_create failed");
        msg_queue_destroy(pJunqi->qid);
        pJunqi->qid = NULL;
        return 0;
    }
    return tidp;
}
