"""
Setup secure credentials and initialization file for Servy MySQL 8.0.
Generates strong secrets, creates init_bootstrap.sql, updates my.ini.
"""
import os
import secrets
import string

def generate_strong_password(length=24):
    alphabet = string.ascii_letters + string.digits + "_-!#"
    # Ensure at least one lower, upper, digit
    while True:
        pwd = ''.join(secrets.choice(alphabet) for _ in range(length))
        if (any(c.islower() for c in pwd) and
            any(c.isupper() for c in pwd) and
            any(c.isdigit() for c in pwd)):
            return pwd

def main():
    root_password = generate_strong_password(24)
    app_password = generate_strong_password(24)

    # Write init_bootstrap.sql
    init_sql_path = "C:/mysql/init_bootstrap.sql"
    sql_content = f"""-- MySQL 8.0 Initialization Script for Servy RAG
ALTER USER 'root'@'localhost' IDENTIFIED BY '{root_password}';
CREATE USER IF NOT EXISTS 'root'@'127.0.0.1' IDENTIFIED BY '{root_password}';
GRANT ALL PRIVILEGES ON *.* TO 'root'@'127.0.0.1' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON *.* TO 'root'@'localhost' WITH GRANT OPTION;

CREATE DATABASE IF NOT EXISTS servy CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
CREATE DATABASE IF NOT EXISTS test_servy CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;

CREATE USER IF NOT EXISTS 'servy_app'@'localhost' IDENTIFIED BY '{app_password}';
CREATE USER IF NOT EXISTS 'servy_app'@'127.0.0.1' IDENTIFIED BY '{app_password}';

GRANT ALL PRIVILEGES ON servy.* TO 'servy_app'@'localhost';
GRANT ALL PRIVILEGES ON servy.* TO 'servy_app'@'127.0.0.1';
GRANT ALL PRIVILEGES ON test_servy.* TO 'servy_app'@'localhost';
GRANT ALL PRIVILEGES ON test_servy.* TO 'servy_app'@'127.0.0.1';

FLUSH PRIVILEGES;
"""
    with open(init_sql_path, "w", encoding="utf-8") as f:
        f.write(sql_content)
    print(f"Written initialization script to {init_sql_path}")

    # Save credentials locally in a git-ignored file
    creds_path = os.path.abspath(".env.mysql_credentials")
    with open(creds_path, "w", encoding="utf-8") as f:
        f.write(f"MYSQL_ROOT_PASSWORD={root_password}\n")
        f.write(f"MYSQL_HOST=127.0.0.1\n")
        f.write(f"MYSQL_PORT=3306\n")
        f.write(f"MYSQL_DATABASE=servy\n")
        f.write(f"MYSQL_USER=servy_app\n")
        f.write(f"MYSQL_PASSWORD={app_password}\n")
    print(f"Credentials saved securely to {creds_path}")

    # Ensure .env.mysql_credentials is in .gitignore
    gitignore_path = os.path.abspath(".gitignore")
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8") as f:
            lines = f.read()
        if ".env.mysql_credentials" not in lines:
            with open(gitignore_path, "a", encoding="utf-8") as f:
                f.write("\n.env.mysql_credentials\n")
            print("Added .env.mysql_credentials to .gitignore")

    # Update C:/mysql/my.ini to add init-file if not already present
    my_ini_path = "C:/mysql/my.ini"
    with open(my_ini_path, "r", encoding="utf-8") as f:
        my_ini_content = f.read()

    if "init-file=" not in my_ini_content:
        # insert init-file under [mysqld]
        lines = my_ini_content.splitlines()
        new_lines = []
        for line in lines:
            new_lines.append(line)
            if line.strip() == "[mysqld]":
                new_lines.append(f"init-file={init_sql_path}")
        with open(my_ini_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        print(f"Updated {my_ini_path} with init-file")

if __name__ == '__main__':
    main()
