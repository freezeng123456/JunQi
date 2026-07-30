/* Minimal getopt_long compatibility for the MinGW build. */
#ifndef JUNQI_GETOPT_H_
#define JUNQI_GETOPT_H_

#ifdef _WIN32

#include <string.h>

struct option
{
	const char *name;
	int has_arg;
	int *flag;
	int val;
};

#define no_argument       0
#define required_argument 1
#define optional_argument 2

static char *junqi_optarg;
static int junqi_optind = 1;

static int junqi_getopt_long(int argc, char *const argv[], const char *shortopts,
						 const struct option *longopts, int *longindex)
{
	char *arg;
	const char *shortopt;
	int i;

	(void)shortopts;
	if (junqi_optind >= argc)
		return -1;
	arg = argv[junqi_optind];
	if (arg == NULL || arg[0] != '-' || strcmp(arg, "-") == 0)
		return -1;
	if (strcmp(arg, "--") == 0)
	{
		junqi_optind++;
		return -1;
	}

	if (arg[1] == '-')
	{
		const char *name = arg + 2;
		const char *equals = strchr(name, '=');
		size_t name_len = equals ? (size_t)(equals - name) : strlen(name);
		for (i = 0; longopts != NULL && longopts[i].name != NULL; i++)
		{
			if (strlen(longopts[i].name) != name_len ||
				strncmp(longopts[i].name, name, name_len) != 0)
				continue;
			if (longindex != NULL)
				*longindex = i;
			if (longopts[i].has_arg == required_argument)
			{
				if (equals != NULL)
					junqi_optarg = (char *)equals + 1;
				else if (junqi_optind + 1 < argc)
					junqi_optarg = argv[++junqi_optind];
				else
				{
					junqi_optind++;
					return '?';
				}
			}
			else
				junqi_optarg = NULL;
			junqi_optind++;
			return longopts[i].val;
		}
		junqi_optind++;
		return '?';
	}

	shortopt = strchr(shortopts, arg[1]);
	if (shortopt == NULL)
	{
		junqi_optind++;
		return '?';
	}
	if (shortopt[1] == ':')
	{
		if (arg[2] != '\0')
			junqi_optarg = arg + 2;
		else if (junqi_optind + 1 < argc)
			junqi_optarg = argv[++junqi_optind];
		else
		{
			junqi_optind++;
			return '?';
		}
	}
	else
		junqi_optarg = NULL;
	junqi_optind++;
	return (unsigned char)arg[1];
}

#define getopt_long  junqi_getopt_long
#define optarg       junqi_optarg
#define optind       junqi_optind

#else

#include <getopt.h>

#endif

#endif /* JUNQI_GETOPT_H_ */
