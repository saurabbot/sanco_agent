from db.db import DatabaseManager
from typing import Optional, List, Any, Dict
from datetime import datetime
import uuid

class CRUD:
    def __init__(self) -> None:
        self.db_manager = DatabaseManager()

    async def create_chat_session(
        self,
        session_uuid: str = None,
        user_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        platform: Optional[str] = None,
        session_type: str = "chat"
    ) -> Dict[str, Any]:
        if not session_uuid:
            session_uuid = str(uuid.uuid4())
            
        query = """
            INSERT INTO chatsession (
                session_uuid, user_id, ip_address, user_agent, platform, session_type, created_at, last_activity_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING *
        """
        now = datetime.now()
        result = await self.db_manager.fetchrow(
            query, session_uuid, user_id, ip_address, user_agent, platform, session_type, now, now
        )
        return dict(result)

    async def get_chat_session_by_uuid(self, session_uuid: str) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM chatsession WHERE session_uuid = $1"
        result = await self.db_manager.fetchrow(query, session_uuid)
        return dict(result) if result else None

    async def update_last_activity(self, session_id: int) -> None:
        query = "UPDATE chatsession SET last_activity_at = $1 WHERE id = $2"
        await self.db_manager.execute(query, datetime.now(), session_id)

    async def create_transcription(
        self,
        session_id: int,
        transcription: str,
        start_time: datetime,
        end_time: datetime,
        transcription_summary: Optional[str] = None
    ) -> Dict[str, Any]:
        query = """
            INSERT INTO transcription (
                session_id, transcription, start_time, end_time, transcription_summary, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING *
        """
        now = datetime.now()
        result = await self.db_manager.fetchrow(
            query, session_id, transcription, start_time, end_time, transcription_summary, now, now
        )
        return dict(result)

    async def update_transcription_summary(self, transcription_id: int, summary: str) -> None:
        query = "UPDATE transcription SET transcription_summary = $1, updated_at = $2 WHERE id = $3"
        await self.db_manager.execute(query, summary, datetime.now(), transcription_id)

    async def get_transcriptions_for_session(self, session_id: int) -> List[Dict[str, Any]]:
        query = "SELECT * FROM transcription WHERE session_id = $1 ORDER BY created_at ASC"
        records = await self.db_manager.fetch(query, session_id)
        return [dict(r) for r in records]
