from typing import Optional, List, Any, Dict
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    def __init__(self) -> None:
        self.connection_string = None
        self.pool = None

    def _ensure_connection_string(self):
        if not self.connection_string:
            self.connection_string = os.getenv("DATABASE_URL")
            if not self.connection_string:
                raise ValueError("DATABASE_URL environment variable is required")

    async def connect(self):
        if not self.pool:
            self._ensure_connection_string()
            self.pool = await asyncpg.create_pool(
                self.connection_string, min_size=1, max_size=10, command_timeout=60
            )

    async def disconnect(self):
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def execute(self, query: str, *args: Any) -> Any:
        if not self.pool:
            await self.connect()
        async with self.pool.acquire() as connection:
            return await connection.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> List[asyncpg.Record]:
        if not self.pool:
            await self.connect()
        async with self.pool.acquire() as connection:
            return await connection.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> Optional[asyncpg.Record]:
        if not self.pool:
            await self.connect()
        async with self.pool.acquire() as connection:
            return await connection.fetchrow(query, *args)
