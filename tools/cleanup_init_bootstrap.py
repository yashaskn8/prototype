import os

# Remove init-file from my.ini
my_ini_path = "C:/mysql/my.ini"
with open(my_ini_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = [l for l in lines if not l.strip().startswith("init-file=")]

with open(my_ini_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

# Remove init_bootstrap.sql
init_sql = "C:/mysql/init_bootstrap.sql"
if os.path.exists(init_sql):
    os.remove(init_sql)
    print(f"Removed temporary bootstrap script {init_sql}")

print("Cleaned up initialization file from my.ini and disk.")
