#!/usr/bin/sh

#SBATCH --job-name=DATABASE-CH0
#SBATCH --time=48:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --dependency=singleton
#SBATCH --output=./slurm_output/database-%j.out

env | grep "^SLURM" | sort

PORT=5433
DB_DIR="/hpc/mydata/$USER/postgresql_ultrack"
DB_NAME="ultrack"
DB_SOCKET_DIR="/tmp"

# fixing error "FATAL:  unsupported frontend protocol 1234.5679: server supports 2.0 to 3.0"
# reference: https://stackoverflow.com/questions/59190010/psycopg2-operationalerror-fatal-unsupported-frontend-protocol-1234-5679-serve
DB_ADDR="$USER:$ULTRACK_DB_PW@$SLURM_JOB_NODELIST:$PORT/ultrack?gssencmode=disable"

# update config file
echo ""
echo "Server running on uri $DB_ADDR"
dasel put string -f $CFG_FILE "data.address" $DB_ADDR
echo "Updated $CFG_FILE"

rm -r $DB_DIR
mkdir -p $DB_DIR
initdb $DB_DIR

# setting new lock directory
cat << EOF >> $DB_DIR/postgresql.conf
unix_socket_directories = '$DB_SOCKET_DIR'
EOF

# creating database
pg_ctl start -D $DB_DIR

# allowing $USER network access
cat << EOF >> $DB_DIR/pg_hba.conf
host    all             $USER           samenet                 md5
EOF

createdb -h $DB_SOCKET_DIR $DB_NAME

# updating user password
psql -h $DB_SOCKET_DIR -c "ALTER USER \"$USER\" PASSWORD '$ULTRACK_DB_PW';" $DB_NAME

# turn on logging
psql -h $DB_SOCKET_DIR -c "ALTER SYSTEM SET logging_collector TO 'on';" $DB_NAME

# configuration tuned using https://pgtune.leopard.in.ua/
# and SLURM job parameters
# -- WARNING
# -- this tool not being optimal
# -- for very high memory systems
# -- DB Version: 10
# -- OS Type: linux
# -- DB Type: dw
# -- Total Memory (RAM): 128 GB
# -- CPUs num: 16
# -- Connections num: 500
# -- Data Storage: hdd
psql -h $DB_SOCKET_DIR -c "ALTER SYSTEM SET max_connections = '500';" $DB_NAME
psql -h $DB_SOCKET_DIR -c "ALTER SYSTEM SET shared_buffers = '32GB';" $DB_NAME
psql -h $DB_SOCKET_DIR -c "ALTER SYSTEM SET effective_cache_size = '96GB';" $DB_NAME
psql -h $DB_SOCKET_DIR -c "ALTER SYSTEM SET maintenance_work_mem = '2GB';" $DB_NAME
psql -h $DB_SOCKET_DIR -c "ALTER SYSTEM SET checkpoint_completion_target = '0.9';" $DB_NAME
psql -h $DB_SOCKET_DIR -c "ALTER SYSTEM SET wal_buffers = '16MB';" $DB_NAME
psql -h $DB_SOCKET_DIR -c "ALTER SYSTEM SET default_statistics_target = '500';" $DB_NAME
psql -h $DB_SOCKET_DIR -c "ALTER SYSTEM SET random_page_cost = '4';" $DB_NAME
psql -h $DB_SOCKET_DIR -c "ALTER SYSTEM SET effective_io_concurrency = '2';" $DB_NAME
psql -h $DB_SOCKET_DIR -c "ALTER SYSTEM SET work_mem = '4194kB';" $DB_NAME
psql -h $DB_SOCKET_DIR -c "ALTER SYSTEM SET huge_pages = 'try';" $DB_NAME
psql -h $DB_SOCKET_DIR -c "ALTER SYSTEM SET min_wal_size = '4GB';" $DB_NAME
psql -h $DB_SOCKET_DIR -c "ALTER SYSTEM SET max_wal_size = '16GB';" $DB_NAME
psql -h $DB_SOCKET_DIR -c "ALTER SYSTEM SET max_worker_processes = '16';" $DB_NAME
psql -h $DB_SOCKET_DIR -c "ALTER SYSTEM SET max_parallel_workers_per_gather = '8';" $DB_NAME
psql -h $DB_SOCKET_DIR -c "ALTER SYSTEM SET max_parallel_workers = '16';" $DB_NAME

# restart database
pg_ctl stop -D $DB_DIR
postgres -i -D $DB_DIR -p $PORT

# STOP:
# pg_ctl stop -D $DB_DIR -p $PORT

# DUMP
# pg_dump -f data.sql -d ultrack -h <NODE> -p <PORT> -U $USER
