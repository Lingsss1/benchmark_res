# Security policy

## Supported versions

Security fixes are provided for the latest tagged release.

## Reporting a vulnerability

Do not open a public issue for sandbox escapes, secret exposure, hidden-label
leakage or arbitrary host-code execution. Contact the maintainers privately
before disclosure. Until a project security address is published, use GitHub's
private vulnerability reporting feature on the repository.

Include the affected version, reproduction steps, expected impact and any
suggested mitigation. Maintainers should acknowledge a report within seven
days and coordinate a disclosure timeline with the reporter.

## Threat model

Evolution agents and candidate workspaces are untrusted code. Official runs
must use `run-agent-docker` or an equivalent audited sandbox. The local runner
is only for trusted development. Never expose host credentials, Docker sockets,
cloud metadata endpoints, private datasets or grader source to an agent.

