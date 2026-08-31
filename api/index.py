# Vercel's classic file-based Python function convention: any .py file under
# /api becomes a Vercel Function. Logic lives in api_server.py at the project
# root; this just re-exports its ASGI app.
from api_server import app
