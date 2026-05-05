from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.services.legal_assistant_service import LegalAssistantService


@lru_cache(maxsize=1)
def get_legal_assistant_service() -> "LegalAssistantService":
    from src.services.legal_assistant_service import LegalAssistantService

    return LegalAssistantService()
