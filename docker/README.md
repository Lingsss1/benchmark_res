# Sandboxed evaluation

Build the reference image:

```bash
docker build -t recevobench:local -f docker/Dockerfile .
```

`recevobench.sandbox` launches candidate code with:

- no network;
- a read-only root filesystem;
- all Linux capabilities dropped and `no-new-privileges` enabled;
- bounded memory, CPU processes and wall time;
- the candidate workspace and task definition mounted read-only;
- public data mounted for feedback, with private data mounted only for trusted grading;
- a dedicated writable output mount.

The local non-Docker executor is intended for development with trusted code. Official scores must use the Docker executor or an equivalent externally audited sandbox. API keys and host credentials must never be mounted into a candidate container.

