##
## This is to interrogate SQLite directly but you can use Chroma library instead 
##
import sqlite3

dbfile = "./chroma/chroma.sqlite3"
conn = sqlite3.connect(dbfile)
cursor = conn.cursor()
table_list = [a for a in cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]

for table in table_list:
    column_list = [a for a in cursor.execute("PRAGMA table_info(" + table[0] + ")")]
    print("\n" + table[0])
    for column in column_list:
        print(column)

conn.close()