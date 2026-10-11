import sqlite3

c = sqlite3.connect("data/ats_state.sqlite3")
print(c.execute("SELECT name FROM sqlite_master").fetchall())
