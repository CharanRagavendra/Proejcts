import sqlite3 
connection = sqlite3.connect('orders.db') 

connection.execute('''CREATE TABLE ord1 
        (User_ID INT PRIMARY KEY     NOT NULL,
         Order_ID INT UNIQUE NOT NULL,
         Order_Name Varchar(50) NOT NULL, 
         Desription Varchar(50) NOT NULL, 
         Order_date DATE ,
         Delivery_Date DATE,
        Status Char(50))             
         ;''') 
  
# close the connection 
connection.close() 