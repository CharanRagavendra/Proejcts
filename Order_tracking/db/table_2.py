import sqlite3 
connection = sqlite3.connect('orders.db') 

connection.execute('''CREATE TABLE ord2 
         ( Order_ID INT NOT NULL,
         TransitionFrom char(50),
         TransitionTo char(50),
         TransitionDate DATE )             
         ;''') 
  
# close the connection 
connection.close() 