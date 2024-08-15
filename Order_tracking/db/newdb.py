import sqlite3 
connection = sqlite3.connect('orders1.db')
cursor = connection.cursor();

connection.execute('''CREATE TABLE ord2 
         ( Order_ID INT NOT NULL,
         TransitionFrom char(50),
         TransitionTo char(50),
         TransitionDate DATE )             
         ;''') 
  
cursor.execute('''Insert into ord2(Order_ID ,TransitionFrom ,TransitionTo  ,TransitionDate) Values(100001,'mumbai','bangalore','2024-05-02');''')
cursor.execute('''Insert into ord2(Order_ID ,TransitionFrom ,TransitionTo  ,TransitionDate) Values(100001,'mumbai','bangalore','2024-05-03');''')
cursor.execute('''Insert into ord2(Order_ID ,TransitionFrom ,TransitionTo  ,TransitionDate) Values(100002,'Kolkata','Hydrabadh','2024-06-02');''')
cursor.execute('''Insert into ord2(Order_ID ,TransitionFrom ,TransitionTo  ,TransitionDate) Values(100002,'Hydrabadh','Chennai','2024-05-03');''')
cursor.execute('''Insert into ord2(Order_ID ,TransitionFrom ,TransitionTo  ,TransitionDate) Values(100002,'Chennai','Madurai','2024-05-04');''')


connection.commit();




connection.close() 