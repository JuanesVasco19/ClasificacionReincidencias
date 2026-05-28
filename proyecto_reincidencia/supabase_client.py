import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY", "")
        if not url or not key:
            raise EnvironmentError(
                "SUPABASE_URL y SUPABASE_KEY no están definidos. "
                "Crea un archivo .env con esas variables (ver .env.example)."
            )
        _client = create_client(url, key)
    return _client
