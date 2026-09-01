from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 5