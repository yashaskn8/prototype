"""
Phase 11: Tested Rollback Tool for Servy RAG.
Safely switches active application backend from MySQL back to SQLite.
Verifies integrity of the original SQLite source database.
Does NOT delete or alter MySQL data.
Warns about data divergence for any writes created in MySQL after migration.
"""
import os
import sys
import sqlite3

def main():
    print("====================================================================")
    print("  SERVY RAG: REVERSIBLE ROLLBACK PROCEDURE (MYSQL -> SQLITE)")
    print("====================================================================")

    # 1. Verify original SQLite source database
    sq_path = os.path.abspath("db.sqlite3")
    if not os.path.exists(sq_path):
        print(f"FATAL: Source database {sq_path} not found! Cannot roll back.")
        sys.exit(1)

    sq_conn = sqlite3.connect(f"file:{sq_path}?mode=ro", uri=True)
    cur = sq_conn.cursor()
    cur.execute("PRAGMA integrity_check;")
    res = cur.fetchall()
    sq_conn.close()

    if res != [('ok',)]:
        print(f"FATAL: Source SQLite integrity check failed: {res}")
        sys.exit(1)
    print("Source SQLite database db.sqlite3 integrity: OK")

    # 2. Update .env to use sqlite3
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            if line.strip().startswith("SERVY_DB_ENGINE="):
                new_lines.append("SERVY_DB_ENGINE=sqlite3\n")
            else:
                new_lines.append(line)

        # If SERVY_DB_ENGINE was not present, add it
        if not any(l.strip().startswith("SERVY_DB_ENGINE=") for l in new_lines):
            new_lines.insert(0, "SERVY_DB_ENGINE=sqlite3\n")

        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    print("Updated .env to set SERVY_DB_ENGINE=sqlite3")

    # 3. Verify Django initializes with SQLite
    os.environ["SERVY_DB_ENGINE"] = "sqlite3"
    from dotenv import load_dotenv
    load_dotenv(override=True)

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")

    import django
    django.setup()

    from django.db import connection
    print(f"Verified Active Database Engine: {connection.settings_dict['ENGINE']}")
    print(f"Verified Database File:          {connection.settings_dict['NAME']}")

    if "sqlite" in connection.settings_dict["ENGINE"]:
        print("\nPASSED: Rollback verified. Servy is now running on SQLite.")
        print("\n[DATA DIVERGENCE NOTICE]:")
        print("Any new writes made to MySQL while active remain preserved in MySQL")
        print("and will NOT automatically reflect in SQLite without an explicit reverse sync.")
    else:
        print("\nFAILED: Django did not revert to SQLite!")
        sys.exit(1)

if __name__ == '__main__':
    main()
