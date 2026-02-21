import sqlite3
conn = sqlite3.connect(r'c:\Users\jhj20\Desktop\project\EBRCS\data\ebrcs.db')
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print('Tables:', [t[0] for t in tables])
print('Products:', conn.execute('SELECT COUNT(*) FROM products').fetchone()[0])
print('Prices:', conn.execute('SELECT COUNT(*) FROM product_prices').fetchone()[0])
try:
    print('Users:', conn.execute('SELECT COUNT(*) FROM users').fetchone()[0])
except Exception as e:
    print(f'Users table: MISSING ({e})')
try:
    print('Purchase history:', conn.execute('SELECT COUNT(*) FROM purchase_history').fetchone()[0])
except Exception as e:
    print(f'Purchase history: MISSING ({e})')
conn.close()
