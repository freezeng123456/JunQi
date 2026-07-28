# Security policy

## Credentials

Never commit access tokens, passwords, private keys, authenticated remote URLs,
or cloud credentials. Use the operating-system credential manager or a
short-lived environment variable.

If a credential is pasted into a chat, terminal command, log, or commit, revoke
and rotate it immediately. Removing a credential from the latest file does not
remove it from Git history.

## Private artifacts

The repository is proprietary. Training checkpoints, private game datasets,
runtime logs, generated visual experiments, and machine-specific launch files
must remain outside version control unless a maintainer explicitly approves
them.

## Reporting

Report a vulnerability privately to the repository owner. Include the affected
version, reproduction steps, impact, and any suggested mitigation. Do not open
a public issue containing exploitable details or credentials.
