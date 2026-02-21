import sqlite3

DB_PATH = r"c:\Users\jhj20\Desktop\project\EBRCS\data\ebrcs.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

tables = [
    r[0]
    for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
]

print("DB_PATH:", DB_PATH)
print("Tables:", tables)

for table in tables:
    print("\n[{}]".format(table))
    cols = cur.execute("PRAGMA table_info({})".format(table)).fetchall()
    for c in cols:
        print(" ", c)
    idx = cur.execute("PRAGMA index_list({})".format(table)).fetchall()
    if idx:
        print("  indexes:")
        for i in idx:
            print("   ", i)

conn.close()
