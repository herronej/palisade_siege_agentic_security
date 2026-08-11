#!/bin/bash
# PALISADE test controller.
#
# Two configurations from one image, selected by PALISADE_ACCOUNTING:
#
#   0 (default)  no accounting database. The controller can check only whether a
#                named object exists on the cluster; it cannot validate an
#                account, an association or a QoS at all. This is the original
#                configuration and it reproduces unchanged.
#   1            slurmdbd + mariadb with AccountingStorageEnforce, so the
#                association check runs at submission (R13-10).
#
# The association set is supplied rather than hard-coded, because which accounts
# a facility has provisioned is exactly the variable under test:
#
#   PALISADE_ACCOUNTS   comma-separated accounts to create (default: msr_thermo,
#                      which is the sole allocation in G5's own policy object)
#   PALISADE_QOS        comma-separated QoS to create (default: normal,high).
#                      `premium` must NOT appear here -- it is the injected
#                      value the b5_11 survivor carries.
H=$(hostname)
sed "s/__HOST__/$H/g" /etc/slurm/slurm.conf.tmpl > /etc/slurm/slurm.conf
install -d -o munge -g munge -m 0755 /run/munge
install -d -o munge -g munge -m 0700 /var/log/munge /var/lib/munge
install -d -o slurm -g slurm -m 0755 /var/spool/slurmctld /var/spool/slurmd /var/log/slurm
[ -f /etc/munge/munge.key ] || dd if=/dev/urandom bs=1024 count=1 of=/etc/munge/munge.key 2>/dev/null
chown munge:munge /etc/munge/munge.key; chmod 400 /etc/munge/munge.key
runuser -u munge -- /usr/sbin/munged; sleep 1

if [ "${PALISADE_ACCOUNTING:-0}" = "1" ]; then
  ACCOUNTS="${PALISADE_ACCOUNTS:-msr_thermo}"
  QOSLIST="${PALISADE_QOS:-normal,high}"

  install -d -o mysql -g mysql -m 0755 /run/mysqld
  mysqld_safe --skip-grant-tables=0 >/var/log/mysqld.out 2>&1 &
  for _ in $(seq 1 40); do mysqladmin ping >/dev/null 2>&1 && break; sleep 1; done
  mysql -e "CREATE DATABASE IF NOT EXISTS slurm_acct_db;
            CREATE USER IF NOT EXISTS 'slurm'@'localhost' IDENTIFIED BY 'palisade';
            GRANT ALL ON slurm_acct_db.* TO 'slurm'@'localhost';
            FLUSH PRIVILEGES;"

  install -d -o slurm -g slurm -m 0755 /run/slurm
  install -o slurm -g slurm -m 0600 /etc/slurm/slurmdbd.conf.tmpl /etc/slurm/slurmdbd.conf
  # slurmdbd takes no -f: it reads /etc/slurm/slurmdbd.conf, and passing one
  # makes it print usage and exit, which leaves slurmctld to die on "slurmdbd
  # and/or database must be up at slurmctld start time".
  runuser -u slurm -- /usr/sbin/slurmdbd
  for _ in $(seq 1 40); do
    sacctmgr -n show cluster >/dev/null 2>&1 && break; sleep 1
  done

  # Enforcement must be live before slurmctld starts, or the controller comes up
  # with no association check and the whole run measures the old configuration.
  cat >> /etc/slurm/slurm.conf <<EOF
AccountingStorageType=accounting_storage/slurmdbd
AccountingStorageHost=localhost
AccountingStoragePort=6819
AccountingStorageEnforce=associations,qos
EOF
fi

runuser -u slurm -- /usr/sbin/slurmctld -f /etc/slurm/slurm.conf; sleep 3
/usr/sbin/slurmd -f /etc/slurm/slurm.conf; sleep 3
scontrol update NodeName=$H State=RESUME 2>/dev/null || true
scontrol update NodeName=$H State=IDLE 2>/dev/null || true

if [ "${PALISADE_ACCOUNTING:-0}" = "1" ]; then
  sacctmgr -i add cluster palisade 2>/dev/null
  sleep 2
  IFS=',' read -ra QS <<< "$QOSLIST"
  for q in "${QS[@]}"; do sacctmgr -i add qos "$q" 2>/dev/null; done
  ALLQOS=$(IFS=,; echo "${QS[*]}")
  IFS=',' read -ra AS <<< "$ACCOUNTS"
  for a in "${AS[@]}"; do
    sacctmgr -i add account "$a" cluster=palisade \
      Description="palisade test allocation" Organization=ornl 2>/dev/null
    sacctmgr -i modify account "$a" set qos="$ALLQOS" 2>/dev/null
  done
  # The sweep submits over `docker exec`, i.e. as root, so root is the principal
  # that needs the association. AdminLevel is left at None deliberately: an
  # Operator or Administrator would bypass the very check under test.
  DEF="${AS[0]}"
  for a in "${AS[@]}"; do
    sacctmgr -i add user root account="$a" cluster=palisade \
      defaultaccount="$DEF" adminlevel=None 2>/dev/null
  done
  sacctmgr -i modify user root set qos="$ALLQOS" 2>/dev/null
  scontrol reconfigure 2>/dev/null || true
  sleep 2
  echo "[entrypoint] accounting on: accounts=$ACCOUNTS qos=$QOSLIST"
  sacctmgr -n -P show assoc format=Cluster,Account,User,QOS 2>/dev/null
fi

tail -f /dev/null
