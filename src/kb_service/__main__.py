from kb_service.settings import Settings
from kb_service.app import create_app
import uvicorn

if __name__ == "__main__":
    settings = Settings.load()
    uvicorn.run(create_app(), host=settings.host, port=settings.port)
