#!/usr/bin/sh

#SBATCH --job-name=DATABASE
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=10
#SBATCH --dependency=singleton
#SBATCH --output=./output/database-%j.out 

env | grep "^SLURM" | sort

DB_DIR="/hpc/mydata/$USER/postgresql_ultrack"
DB_NAME="ultrack"
DB_SOCKET_DIR="/tmp"

# fixing error "FATAL:  unsupported frontend protocol 1234.5679: server supports 2.0 to 3.0"
# reference: https://stackoverflow.com/questions/59190010/psycopg2-operationalerror-fatal-unsupported-frontend-protocol-1234-5679-serve
DB_ADDR="$USER:$ULTRACK_DB_PW@$SLURM_JOB_NODELIST:5432/ultrack?gssencmode=disable"

# update config file
echo ""
echo "Server running on uri $DB_ADDR"
srun dasel put string -f $CFG_FILE "data.address" $DB_ADDR
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

# increasing max connections
psql -h $DB_SOCKET_DIR -c "ALTER SYSTEM SET max_connections TO '500';" $DB_NAME

# turn on logging
psql -h $DB_SOCKET_DIR -c "ALTER SYSTEM SET logging_collector TO 'on';" $DB_NAME

# increases WAL size to improve performance
# https://www.postgresql.org/docs/current/wal-configuration.html
psql -h $DB_SOCKET_DIR -c "ALTER SYSTEM SET max_wal_size TO '10GB';" $DB_NAME

# restart database
pg_ctl stop -D $DB_DIR
srun postgres -i -D $DB_DIR

# STOP:
# pg_ctl stop -D $DB_DIR

