import sqlite3 
connection = sqlite3.connect('orders.db')
cursor = connection.cursor();
#cursor.execute('''Insert into ord1(User_ID ,Order_ID ,Order_Name ,Desription ,Order_date ,Delivery_Date ,Status  ) Values(000001, 000001, 'Toys', 'toy car', '2024-06-06','2024-06-08','Not Deliverd');''')
cursor.execute('''Insert into ord1(User_ID ,Order_ID ,Order_Name ,Desription ,Order_date ,Delivery_Date ,Status  ) Values(100001, 100001, 'Utensils', 'Spoon', '2024-05-01','2024-05-03','Deliverd');''')
cursor.execute('''Insert into ord1(User_ID ,Order_ID ,Order_Name ,Desription ,Order_date ,Delivery_Date ,Status  ) Values(100002, 100002, 'Electronic', 'Battery', '2024-06-02','2024-06-04','Deliverd');''')
cursor.execute('''Insert into ord1(User_ID ,Order_ID ,Order_Name ,Desription ,Order_date ,Delivery_Date ,Status  ) Values(100003, 100003, 'Cleaning', 'Wet wipes', '2024-06-06','2024-06-10','Not Deliverd');''')
cursor.execute('''Insert into ord1(User_ID ,Order_ID ,Order_Name ,Desription ,Order_date ,Delivery_Date ,Status  ) Values(100004, 100004, 'Furniture', 'Couch', '2024-06-06','2024-06-12','Not Deliverd');''')


connection.commit();




connection.close() 