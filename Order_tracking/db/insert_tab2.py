import sqlite3 
connection = sqlite3.connect('orders.db')
cursor = connection.cursor();
cursor.execute('''Insert into ord2(Order_ID ,TransitionFrom ,TransitionTo  ,TransitionDate) Values(100001,'mumbai','bangalore','2024-05-02');''')
cursor.execute('''Insert into ord2(Order_ID ,TransitionFrom ,TransitionTo  ,TransitionDate) Values(100001,'mumbai','bangalore','2024-05-03');''')
cursor.execute('''Insert into ord2(Order_ID ,TransitionFrom ,TransitionTo  ,TransitionDate) Values(100002,'Kolkata','Hydrabadh','2024-06-02');''')
cursor.execute('''Insert into ord2(Order_ID ,TransitionFrom ,TransitionTo  ,TransitionDate) Values(100002,'Hydrabadh','Chennai','2024-05-03');''')
cursor.execute('''Insert into ord2(Order_ID ,TransitionFrom ,TransitionTo  ,TransitionDate) Values(100002,'Chennai','Madurai','2024-05-04');''')


connection.commit();




connection.close() 