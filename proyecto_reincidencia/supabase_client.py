import os
from dotenv import load_dotenv
import streamlit as st
from supabase import create_client, Client

load_dotenv()

_client: Client | None = None


def _get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, "")


def get_client() -> Client:
    global _client
    if _client is None:
        url = _get_secret("SUPABASE_URL")
        key = _get_secret("SUPABASE_KEY")
        if not url or not key:
            raise EnvironmentError(
                "SUPABASE_URL y SUPABASE_KEY no están definidos. "
                "En local: crea un archivo .env. "
                "En Streamlit Cloud: agrégalos en Settings → Secrets."
            )
        _client = create_client(url, key)
    return _client
