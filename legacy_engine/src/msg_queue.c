#include "msg_queue.h"
#include <stdio.h>

/* Thread-safe producer/consumer queue.
 *
 * Hardening for RL training:
 *   - All malloc return values are checked.
 *   - Queue depth is bounded (MSG_QUEUE_MAX) to prevent unbounded
 *     growth if a consumer stalls (e.g. during a long search tree).
 *   - Overflow drops the oldest message to keep recent state fresh. */

#ifndef MSG_QUEUE_MAX
#define MSG_QUEUE_MAX  4096
#endif

MsgQueue* msg_queue_create() {
    MsgQueue *q = (MsgQueue*)malloc(sizeof(MsgQueue));
    if (q) {
        q->head = NULL;
        q->tail = NULL;
        pthread_mutex_init(&q->mutex, NULL);
        pthread_cond_init(&q->cond, NULL);
        q->count = 0;
    }
    return q;
}

void msg_queue_destroy(MsgQueue *q) {
    if (!q) return;
    pthread_mutex_lock(&q->mutex);
    MsgNode *current = q->head;
    while (current) {
        MsgNode *next = current->next;
        free(current->data);
        free(current);
        current = next;
    }
    pthread_mutex_unlock(&q->mutex);
    pthread_mutex_destroy(&q->mutex);
    pthread_cond_destroy(&q->cond);
    free(q);
}

void msg_queue_send(MsgQueue *q, void *data, size_t len) {
    if (!q || !data || len == 0) return;

    MsgNode *node = (MsgNode*)malloc(sizeof(MsgNode));
    if (!node) {
        /* Allocation failure: drop the message rather than crash. */
        fprintf(stderr, "[msg_queue] malloc node failed, dropping msg\n");
        return;
    }
    node->data = malloc(len);
    if (!node->data) {
        fprintf(stderr, "[msg_queue] malloc data failed (len=%zu), dropping msg\n", len);
        free(node);
        return;
    }
    memcpy(node->data, data, len);
    node->len = len;
    node->next = NULL;

    pthread_mutex_lock(&q->mutex);
    /* Bounded queue: drop oldest when over capacity. */
    if (q->count >= MSG_QUEUE_MAX) {
        MsgNode *drop = q->head;
        if (drop) {
            q->head = drop->next;
            if (!q->head) q->tail = NULL;
            q->count--;
            free(drop->data);
            free(drop);
        }
    }
    if (q->tail) {
        q->tail->next = node;
        q->tail = node;
    } else {
        q->head = node;
        q->tail = node;
    }
    q->count++;
    pthread_cond_signal(&q->cond);
    pthread_mutex_unlock(&q->mutex);
}

size_t msg_queue_receive(MsgQueue *q, void *buffer, size_t max_len) {
    if (!q) return 0;
    pthread_mutex_lock(&q->mutex);
    while (q->count == 0) {
        pthread_cond_wait(&q->cond, &q->mutex);
    }

    MsgNode *node = q->head;
    q->head = node->next;
    if (q->head == NULL) {
        q->tail = NULL;
    }
    q->count--;
    pthread_mutex_unlock(&q->mutex);

    size_t copy_len = node->len > max_len ? max_len : node->len;
    memcpy(buffer, node->data, copy_len);

    free(node->data);
    free(node);

    return copy_len;
}
