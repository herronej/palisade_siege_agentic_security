FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
# slurmdbd + mariadb serve the accounting-enforcement configuration (R13-10).
# They stay inert when CITADEL_ACCOUNTING=0, which is the default and reproduces
# the original no-accounting controller byte for byte.
RUN apt-get update && apt-get install -y --no-install-recommends \
      slurm-wlm munge slurm-client util-linux \
      slurmdbd mariadb-server \
    && rm -rf /var/lib/apt/lists/*
COPY slurm.conf /etc/slurm/slurm.conf.tmpl
COPY slurmdbd.conf /etc/slurm/slurmdbd.conf.tmpl
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
