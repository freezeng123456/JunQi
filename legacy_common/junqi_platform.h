/*
 * junqi_platform.h
 *
 * Small portability layer shared by the legacy engine and GTK client.  The
 * game protocol is UDP based, so keeping the socket type and lifecycle in one
 * place prevents the Windows SOCKET (a pointer-sized value) from being
 * accidentally truncated to an int.
 */
#ifndef JUNQI_PLATFORM_H_
#define JUNQI_PLATFORM_H_

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <errno.h>

typedef SOCKET junqi_socket_t;
#define JUNQI_INVALID_SOCKET INVALID_SOCKET
#define junqi_socket_close closesocket
#define junqi_socket_last_error() ((int)WSAGetLastError())

static inline int junqi_socket_init(void)
{
	WSADATA data;
	return WSAStartup(MAKEWORD(2, 2), &data);
}

static inline void junqi_socket_cleanup(void)
{
	WSACleanup();
}

#define junqi_file_read  _read
#define junqi_file_write _write
#define junqi_file_seek  _lseek
#define junqi_file_close _close

#else

#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>

typedef int junqi_socket_t;
#define JUNQI_INVALID_SOCKET (-1)
#define junqi_socket_close close
#define junqi_socket_last_error() (errno)

static inline int junqi_socket_init(void)
{
	return 0;
}

static inline void junqi_socket_cleanup(void)
{
}

#define junqi_file_read  read
#define junqi_file_write write
#define junqi_file_seek  lseek
#define junqi_file_close close

#endif

#endif /* JUNQI_PLATFORM_H_ */
