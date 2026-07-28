/*
 * communication.c
 *
 *  Created on: Aug 15, 2018
 *      Author: Administrator
 */

#include "comm.h"
#include "junqi.h"
#include "engine.h"
#include <stdarg.h>
#include <errno.h>

// Definition of aMagic (declared extern in comm.h)
const u8 aMagic[4] = {0x57, 0x04, 0, 0};

void PacketHeader(CommHeader *header, u8 iDir, u8 eFun)
{
	memset(header, 0, sizeof(CommHeader));
	memcpy(header->aMagic, aMagic, 4);
	//目前该位只在COMM_START标识先手位
	header->iDir = iDir;
	header->eFun = eFun;
}

void SendData(Junqi* pJunqi, CommHeader *header, void *data, int len)
{
	u8 buf[100];
	int length = 0;

	length += sizeof(CommHeader);
	memcpy(buf, header, length);

	memcpy(buf+length, data, len);
	length += len;

	sendto(pJunqi->socket_fd, buf, length, 0,
			(struct sockaddr *)&pJunqi->addr, sizeof(struct sockaddr));
	LOG_TRACE(LOG_CAT_COMM, "send %d bytes", length);
	SafeMemout(buf,length);

}

void SendHeader(Junqi* pJunqi, u8 iDir, u8 eFun)
{
	CommHeader header;
	PacketHeader(&header, iDir, eFun);
	SendData(pJunqi, &header, NULL, 0);
}

void SetRecLineup(Junqi* pJunqi, u8 *data, int iDir)
{
	int i;
	for(i=0; i<30; i++)
	{
		pJunqi->Lineup[iDir][i].type = data[i];
		//assert( pJunqi->Lineup[iDir][i].type!=DARK );
	}

}

void SendMove(Junqi* pJunqi,  BoardChess *pSrc, BoardChess *pDst)
{
	MoveResultData send_data;
	CommHeader header;

	memset(&send_data, 0, sizeof(MoveResultData));
	send_data.src[0] = pSrc->point.x;
	send_data.src[1] = pSrc->point.y;
	send_data.dst[0] = pDst->point.x;
	send_data.dst[1] = pDst->point.y;

	PacketHeader(&header, pJunqi->eTurn, COMM_MOVE);
	SendData(pJunqi, &header, &send_data, sizeof(MoveResultData));
}

void SendEvent(Junqi* pJunqi, int iDir, u8 event)
{
	CommHeader header;
	u8 data = event;
	PacketHeader(&header, iDir, COMM_EVNET);
	SendData(pJunqi, &header, &data, 1);
}

void DealRecData(Junqi* pJunqi, u8 *data, size_t len)
{
	CommHeader *pHead;
	pHead = (CommHeader *)data;
	/* isBoardInit used to be a function-scope static, which meant
	 * a single process could only ever initialise its board once -
	 * even across multiple reconnects. Moved to Junqi so each
	 * instance manages its own state (needed for RL restarts). */

//	struct mq_attr attr;
//	mq_getattr(pJunqi->qid,&attr);

	if( memcmp(pHead->aMagic, aMagic, 4)!=0 )
	{
		return;
	}


	switch(pHead->eFun)
	{
	case COMM_GO:
		pJunqi->bGo = 1;
		pJunqi->bStop = 0;
		LOG_INFO(LOG_CAT_COMM, "recv COMM_GO");
		//mq_send(pJunqi->qid, (char*)data, len, 0);
		break;
	case COMM_STOP:
		pJunqi->bStop = 1;
		LOG_INFO(LOG_CAT_COMM, "recv COMM_STOP");
		break;
	case COMM_ERROR:
		LOG_ERROR(LOG_CAT_COMM, "peer reported COMM_ERROR (ignored, not aborting)");
		/* Previously this was assert(0), which aborted the entire
		 * process. For RL self-play we must never crash on a single
		 * malformed packet - just log it. */
		break;
	case COMM_START:
		pJunqi->preTurn = 1000;
		//更换布阵后重新初始化棋盘
		InitChess(pJunqi, data);
		SendHeader(pJunqi, pHead->iDir, COMM_OK);
		pJunqi->eTurn = pHead->iDir;
		pJunqi->bStart = 1;
		msg_queue_send(pJunqi->qid, (char*)data, len);
		LOG_INFO(LOG_CAT_COMM, "recv COMM_START first=%d", pHead->iDir);
		SafeMemout(data, len);
		break;
	case COMM_READY:
		pJunqi->bStart = 0;
		pJunqi->nRpStep = 0;
		pJunqi->iRpOfst = 0;
		pJunqi->bGo = 1;
		pJunqi->bSearch = 0;
		pthread_mutex_lock(&pJunqi->mutex);
		/* sleep(1) removed: previously used as a crude way to wait
		 * for the search thread to finish; now we rely solely on the
		 * mutex. This cuts >1s off every game-over transition,
		 * critical for RL self-play throughput. */
		CloseEngine(pJunqi->pEngine);
		pthread_mutex_unlock(&pJunqi->mutex);
		SendHeader(pJunqi, pHead->iDir, COMM_READY);
		LOG_INFO(LOG_CAT_COMM, "recv COMM_READY");
		break;
	case COMM_INIT:
		pthread_mutex_lock(&pJunqi->mutex);
		memset(pJunqi->Lineup,0,sizeof(pJunqi->Lineup));
		pJunqi->pEngine = OpenEngine(pJunqi);
		InitLineup(pJunqi, data, pJunqi->isBoardInit);
		InitChess(pJunqi, data);
		pthread_mutex_unlock(&pJunqi->mutex);
		if( !pJunqi->isBoardInit )
		{
			pJunqi->isBoardInit = 1;
			InitBoard(pJunqi);
		}
		LOG_INFO(LOG_CAT_COMM, "recv COMM_INIT len=%zu", len);
		SafeMemout(data, len);
		SendHeader(pJunqi, pHead->iDir, COMM_OK);
		break;
	case COMM_LINEUP:
		data = (u8*)&pHead[1];
		SetRecLineup(pJunqi, data,  pHead->iDir);
		LOG_DEBUG(LOG_CAT_COMM, "recv COMM_LINEUP dir=%d", pHead->iDir);
		SafeMemout(data, len);
		SendHeader(pJunqi, pHead->iDir, COMM_OK);
		break;
	case COMM_REPLAY:
        //在引擎线程中处理
		msg_queue_send(pJunqi->qid, (char*)data, len);
		break;
	case COMM_MOVE:
	case COMM_EVNET:
		pJunqi->bMove = 1;
		LOG_TRACE(LOG_CAT_COMM, "recv move/event dir=%d fn=%d", pHead->iDir, pHead->eFun);
		msg_queue_send(pJunqi->qid, (char*)data, len);
		break;
	default:
		LOG_WARN(LOG_CAT_COMM, "recv unknown eFun=%d from dir=%d", pHead->eFun, pHead->iDir);
		break;
	}
}

void *comm_thread(void *arg)
{
	Junqi* pJunqi = (Junqi*)arg;
	int socket_fd;
	struct sockaddr_in addr,local;
	size_t recvbytes = 0;
	u8 buf[REC_LEN]={0};
	u16 local_port;
	u16 remote_port;

	socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
	if (socket_fd < 0)
	{
		LOG_ERROR(LOG_CAT_COMM, "Create Socket Failed: errno=%d", errno);
		pthread_detach(pthread_self());
		return NULL;
	}

	local.sin_family = AF_INET;
	local.sin_addr.s_addr=INADDR_ANY;

	/* Port configuration precedence:
	 *   1. pJunqi->netCfg.*_port if non-zero (set via CLI or API)
	 *   2. fallback to compile-time default (TEST or release) */
	local_port  = (pJunqi->netCfg.local_port  != 0)
	              ? pJunqi->netCfg.local_port  : DEFAULT_ENGINE_LOCAL_PORT;
	remote_port = (pJunqi->netCfg.remote_port != 0)
	              ? pJunqi->netCfg.remote_port : DEFAULT_ENGINE_REMOTE_PORT;

	local.sin_port = htons(local_port);
    int opt = 1;
    setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

	if(bind(socket_fd, (struct sockaddr *)&local, sizeof(struct sockaddr) )<0)
	{
		LOG_ERROR(LOG_CAT_COMM, "Bind to port %d failed: errno=%d", local_port, errno);
		close(socket_fd);  /* M4: was leaked before */
		pthread_detach(pthread_self());
        return NULL;
	}

	addr.sin_family = AF_INET;
	inet_pton(AF_INET,
	          pJunqi->netCfg.remote_ip[0] ? pJunqi->netCfg.remote_ip : "127.0.0.1",
	          &addr.sin_addr);
	addr.sin_port = htons(remote_port);

	pJunqi->socket_fd = socket_fd;
	pJunqi->addr = addr;

	LOG_INFO(LOG_CAT_COMM, "comm thread up: listen=:%d target=%s:%d seat=%d",
	         local_port,
	         pJunqi->netCfg.remote_ip[0] ? pJunqi->netCfg.remote_ip : "127.0.0.1",
	         remote_port, JQ_ENGINE_DIR(pJunqi));
	SendHeader(pJunqi, JQ_ENGINE_DIR(pJunqi), COMM_READY);

	while(1)
	{
		recvbytes=recvfrom(socket_fd, buf, REC_LEN, 0,NULL ,NULL);
		if ((int)recvbytes <= 0) {
			LOG_WARN(LOG_CAT_COMM, "recvfrom returned %d, errno=%d",
			         (int)recvbytes, errno);
			continue;
		}
		DealRecData(pJunqi, buf, recvbytes);
	}

	/* Unreachable in current design; retained for completeness. */
	close(socket_fd);
	pthread_detach(pthread_self());
	return NULL;
}

pthread_t CreateCommThread(Junqi* pJunqi)
{
    pthread_t tidp;
    pthread_create(&tidp,NULL,(void*)comm_thread,pJunqi);
    return tidp;
}
