import sqlite3
import json

c = sqlite3.connect("data/ats_state.sqlite3")

print("\n--- POLICY STATE ---")
for row in c.execute("SELECT * FROM policy_state"):
    print(row)

print("\n--- POLICY HISTORY (LATEST 10) ---")
for row in c.execute(
    "SELECT * FROM policy_history ORDER BY rowid DESC LIMIT 10"
):
    print(row)

print("\n--- DIAGNOSTICS (LATEST 10) ---")
for row in c.execute(
    "SELECT * FROM diagnostics ORDER BY rowid DESC LIMIT 10"
):
    print(row)

print("\n--- COUNTS ---")
for table in ["forecasts", "outcomes", "policy_history", "diagnostics"]:
    print(table, c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
