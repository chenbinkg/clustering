import pymysql
import sys
import pandas as pd
import time
import configparser
from sqlalchemy import create_engine

hostname = 'fslmysesdadev01'
username = 'hdfsf10w'
password = 'b24RHz2RHpLn'
database = 'emc'

class DB_QUERY:
    'auther: chenbin'\
    'this class is just a thin wrapper for FD Trend Predictor DB transactions. only args are the sql raw string for MySQL server'


    def __init__(self,sqlstring):
        self.sqlstring=sqlstring

    def sqlquery(self):
        try:
            query_start_time = time.time()
            conn = create_engine("mysql+pymysql://{}:{}@{}/{}".format(username, password, hostname, database))
            #conn = sql_engine.raw_connection()
            #conn = MySQLdb.connect(host=hostname, user=username, passwd=password, db=database, charset='utf8')
            df = pd.read_sql(self.sqlstring , conn)
            #conn.close()
            print("DB Query time: ", time.time() - query_start_time)
        except Exception as err:
            sys.stderr.write('ERROR: %sn' % str(err))
            print('ERROR: %sn' % str(err))
            df = 'Query failed!'

        return df



class DB_INSERT:
    

    def __init__(self, tablename, df):
        self.tablename=tablename
        self.df=df

    def insert_to_db(self) :
        'Insert Data to DB'
        try:
            start_time_db = time.time()
            #conn = MySQLdb.connect(host=hostname, user=username, passwd=password, db=database, charset='utf8')
            conn = create_engine("mysql+pymysql://{}:{}@{}/{}".format(username, password, hostname, database))
            #conn = sql_engine.raw_connection()
            self.df.to_sql(con=conn, name=self.tablename, if_exists='append', index=False)
            #conn.close()
            run_outcome = 'insert success!'
            print("Inserted data to ", self.tablename, " database, DB write time: ", time.time()-start_time_db)
        except Exception as err:
            print("Failed to insert data to ", self.tablename)
            print(str(err))
            run_outcome = 'insert failed!'

        return run_outcome



class DB_UPDATE:
    
    def __init__(self, sqlstring, tablename):
        self.sqlstring=sqlstring
        self.tablename=tablename
    def update_db(self):
        """
        Updates a mysql table with the data built in sql string. 
        Param:
            sqlstring:
                SQL string for update
        """
        start_time_db = time.time()
        # Connection
        conn = create_engine("mysql+pymysql://{}:{}@{}/{}".format(username, password, hostname, database))

        try:
            conn.execute(self.sqlstring)
            print("Update string: ", self.sqlstring)
            print("Updated successfully to ", self.tablename, " , DB write time: ", time.time()-start_time_db)
            run_outcome = "Updated Success for table " + self.tablename
        except Exception as err:
            print("Failed to update data to ", self.tablename)
            print(str(err))
            run_outcome = 'update failed!'

        return run_outcome
