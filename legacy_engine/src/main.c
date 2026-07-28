/*
 * main.c
 *
 *  Created on: Aug 17, 2018
 *      Refactored: CLI flags to configure engine seat and log level
 *      at runtime, preparing for RL self-play where one binary may
 *      be launched 4 times, once per seat.
 */

#include "junqi.h"
#include "comm.h"
#include "engine.h"
#include "search.h"
#include <time.h>
#include <signal.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static volatile sig_atomic_t g_shutdown_requested = 0;

static void handle_shutdown(int sig)
{
	g_shutdown_requested = 1;
	/* We cannot call logging helpers from a signal handler safely,
	 * so we write directly with async-signal-safe calls only. */
	const char msg[] = "\n[signal] shutdown requested, cleaning up...\n";
	(void)write(STDERR_FILENO, msg, sizeof(msg) - 1);
	_exit(128 + sig);
}

static void install_signal_handlers(void)
{
	struct sigaction sa;
	memset(&sa, 0, sizeof(sa));
	sa.sa_handler = handle_shutdown;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = 0;
	sigaction(SIGINT,  &sa, NULL);
	sigaction(SIGTERM, &sa, NULL);
	/* Ignore SIGPIPE - socket partners may close abruptly. */
	signal(SIGPIPE, SIG_IGN);
}

static void print_usage(const char *prog)
{
	fprintf(stderr,
		"Usage: %s [options]\n"
		"  -s, --seat <0-3>         Engine seat index (default: compile-time)\n"
		"  -l, --log-level <0-5>    0=NONE 1=ERR 2=WARN 3=INFO 4=DEBUG 5=TRACE\n"
		"  -c, --log-cat <hex>      Log category bitmask (default 0xFFFFFFFF)\n"
		"  -t, --timeout <sec>      Search timeout in seconds (default 5)\n"
		"  -p, --local-port <p>     UDP port to bind (default: 6678 TEST / 5678 release)\n"
		"  -r, --remote-port <p>    UDP port to send to (default: 1234)\n"
		"  -i, --remote-ip <ip>     Remote IP to send to (default: 127.0.0.1)\n"
		"  -S, --seed <n>           Explicit RNG seed (default: time^pid^seat)\n"
		"  -h, --help               Show this help\n",
		prog);
}

int main(int argc, char *argv[])
{
	Junqi *pJunqi;
	pthread_t t1;

	int opt_seat = -1;
	int opt_log  = -1;
	int opt_timeout = -1;
	int opt_local_port  = -1;
	int opt_remote_port = -1;
	const char *opt_remote_ip = NULL;
	unsigned int opt_cat = 0;
	int has_cat = 0;
	long long opt_seed = -1;

	static struct option long_opts[] = {
		{"seat",        required_argument, 0, 's'},
		{"log-level",   required_argument, 0, 'l'},
		{"log-cat",     required_argument, 0, 'c'},
		{"timeout",     required_argument, 0, 't'},
		{"local-port",  required_argument, 0, 'p'},
		{"remote-port", required_argument, 0, 'r'},
		{"remote-ip",   required_argument, 0, 'i'},
		{"seed",        required_argument, 0, 'S'},
		{"help",        no_argument,       0, 'h'},
		{0, 0, 0, 0}
	};

	int c;
	while ((c = getopt_long(argc, argv, "s:l:c:t:p:r:i:S:h", long_opts, NULL)) != -1) {
		switch (c) {
		case 's': opt_seat        = atoi(optarg); break;
		case 'l': opt_log         = atoi(optarg); break;
		case 'c': opt_cat         = (unsigned int)strtoul(optarg, NULL, 0); has_cat = 1; break;
		case 't': opt_timeout     = atoi(optarg); break;
		case 'p': opt_local_port  = atoi(optarg); break;
		case 'r': opt_remote_port = atoi(optarg); break;
		case 'i': opt_remote_ip   = optarg; break;
		case 'S': opt_seed        = strtoll(optarg, NULL, 0); break;
		case 'h':
		default:
			print_usage(argv[0]);
			return (c == 'h') ? 0 : 1;
		}
	}

	setvbuf(stdout, NULL, _IONBF, 0);
	install_signal_handlers();

	pJunqi = JunqiOpen();
	if (pJunqi == NULL)
		return EXIT_FAILURE;

	if (opt_seat >= 0 && opt_seat < 4) {
		pJunqi->iEngineDir = (u8)opt_seat;
		printf("[main] engine seat overridden to %d\n", opt_seat);
	}
	if (opt_log >= 0) {
		extern int g_log_level;
		g_log_level = opt_log;
		printf("[main] log level set to %d\n", opt_log);
	}
	if (has_cat) {
		extern unsigned int g_log_category;
		g_log_category = opt_cat;
		printf("[main] log category mask set to 0x%08X\n", opt_cat);
	}
	if (opt_timeout > 0) {
		g_search_timeout_sec = opt_timeout;
		printf("[main] search timeout set to %ds\n", opt_timeout);
	}
	if (opt_local_port > 0 && opt_local_port < 65536) {
		pJunqi->netCfg.local_port = (u16)opt_local_port;
		printf("[main] local UDP port = %d\n", opt_local_port);
	}
	if (opt_remote_port > 0 && opt_remote_port < 65536) {
		pJunqi->netCfg.remote_port = (u16)opt_remote_port;
		printf("[main] remote UDP port = %d\n", opt_remote_port);
	}
	if (opt_remote_ip) {
		strncpy(pJunqi->netCfg.remote_ip, opt_remote_ip,
		        sizeof(pJunqi->netCfg.remote_ip) - 1);
		printf("[main] remote IP = %s\n", pJunqi->netCfg.remote_ip);
	}
	if (opt_seed >= 0) {
		extern void random_seed(unsigned int);
		random_seed((unsigned int)opt_seed);
		printf("[main] RNG seed = %u (deterministic)\n", (unsigned int)opt_seed);
	}

	if (CreatEngineThread(pJunqi) == 0)
		return EXIT_FAILURE;
	(void)CreatePrintThread(pJunqi);
	t1 = CreatCommThread(pJunqi);
	if (t1 == 0)
		return EXIT_FAILURE;

	pthread_join(t1, NULL);

	return 0;
}
