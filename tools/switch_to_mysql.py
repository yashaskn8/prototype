"""
Phase 8: Safely configure local .env for authoritative MySQL backend.
Preserves existing settings while adding MySQL configuration.
Verifies active connection using Django runtime.
"""
import os
import sys

def main():
    creds = {}
    with open(".env.mysql_credentials", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                creds[k.strip()] = v.strip()

    env_path = ".env"
    existing_lines = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()

    # Filter out any existing SERVY_DB_ENGINE or MYSQL_* lines
    filtered_lines = [
        l for l in existing_lines
        if not any(l.strip().startswith(prefix) for prefix in [
            "SERVY_DB_ENGINE", "MYSQL_HOST", "MYSQL_PORT", "MYSQL_DATABASE",
            "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_CONNECT_TIMEOUT", "MYSQL_CONN_MAX_AGE"
        ])
    ]

    mysql_config = [
        "\n# Authoritative MySQL Database Configuration\n",
        "SERVY_DB_ENGINE=mysql\n",
        f"MYSQL_HOST={creds.get('MYSQL_HOST', '127.0.0.1')}\n",
        f"MYSQL_PORT={creds.get('MYSQL_PORT', '3306')}\n",
        f"MYSQL_DATABASE={creds.get('MYSQL_DATABASE', 'servy')}\n",
        f"MYSQL_USER={creds.get('MYSQL_USER', 'servy_app')}\n",
        f"MYSQL_PASSWORD={creds.get('MYSQL_PASSWORD')}\n",
        "MYSQL_CONNECT_TIMEOUT=10\n",
        "MYSQL_CONN_MAX_AGE=60\n"
    ]

    new_env_content = "".join(filtered_lines).rstrip() + "\n" + "".join(mysql_config)

    with open(env_path, "w", encoding="utf-8") as f:
        f.write(new_env_content)
    print("Updated .env with authoritative MySQL backend configuration.")

    # Now verify Django initializes and uses MySQL without any manual env injection
    # Clear process environment overrides
    os.environ.pop("SERVY_DB_ENGINE", None)
    os.environ.pop("MYSQL_PASSWORD", None)

    # Let django-dotenv / settings load .env
    from dotenv import load_dotenv
    load_dotenv(override=True)

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")

    import django
    django.setup()

    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SELECT DATABASE(), CURRENT_USER(), @@version;")
        db_name, cur_user, ver = cur.fetchone()

    print(f"\nDjango Active Database Verification:")
    print(f"  ENGINE:       {connection.settings_dict['ENGINE']}")
    print(f"  NAME:         {db_name}")
    print(f"  USER:         {cur_user}")
    print(f"  VERSION:      {ver}")
    print(f"  HOST/PORT:    {connection.settings_dict['HOST']}:{connection.settings_dict['PORT']}")

    if "mysql" in connection.settings_dict["ENGINE"] and db_name == "servy":
        print("\nPASSED: Servy is now natively and authoritatively backed by MySQL 8.0!")
    else:
        print("\nFAILED: Django did not resolve to MySQL servy database!")
        sys.exit(1)

if __name__ == '__main__':
    main()
