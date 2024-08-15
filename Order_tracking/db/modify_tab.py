import sqlite3 
connection = sqlite3.connect('orders1.db')
cursor = connection.cursor();
cursor.execute('''UPDATE ord2 SET TransitionTo = 'Chennai' WHERE Order_ID = 100001 AND TransitionDate = '2024-05-03' ;''')



connection.commit();




connection.close() 