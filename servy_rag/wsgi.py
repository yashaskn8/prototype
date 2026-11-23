import os
from django.core.wsgi import get_wsgi_application
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
# Lets core.apps start background model warm-up (SERVY_PRELOAD_MODELS=1) in the
# production application server, not only under runserver.
os.environ.setdefault("SERVY_SERVING_PROCESS", "1")
application = get_wsgi_application()
