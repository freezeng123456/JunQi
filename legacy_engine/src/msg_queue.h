#ifndef MSG_QUEUE_H
#define MSG_QUEUE_H

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include "type.h"

typedef struct MsgNode {
    void *data;
    size_t len;
    struct MsgNode *next;
} MsgNode;

typedef struct {
    MsgNode *head;
    MsgNode *tail;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int count;
} MsgQueue;

MsgQueue* msg_queue_create();
void msg_queue_destroy(MsgQueue *q);
void msg_queue_send(MsgQueue *q, void *data, size_t len);
size_t msg_queue_receive(MsgQueue *q, void *buffer, size_t max_len);

#endif
