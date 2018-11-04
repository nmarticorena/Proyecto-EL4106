#!/usr/bin/env python

import sqlite3

con = sqlite3.connect("/home/nmarticorena/Proyecto-EL4106/db/energy.db")
cursor=con.cursor()

#cursor.execute('''CREATE TABLE ENERGY 
#	(ID STRING PRIMARY KEY NOT NULL,E_0 FLOAT NOT NULL,E_1 FLOAT NOT NULL,E_2 FLOAT NOT NULL,E_3 FLOAT NOT NULL)''' )

selectQuery='''SELECT E_0,E_1,E_2,E_3 FROM ENERGY where ID=?'''


donde=("[1,1,1,0]",)
cursor.execute(selectQuery,donde)

for i in cursor:
	print(i[0])
	print(i[1])
	print(i[2])
	print(i[3])

