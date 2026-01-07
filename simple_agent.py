import logging
import asyncio
import os
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    JobContext,
    JobRequest,
    JobProcess,
    AgentSession,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import silero, deepgram, openai
from livekit.plugins.turn_detector.english import EnglishModel

logger = logging.getLogger("simple_car_vendor_agent")
load_dotenv()
agent_display_name = "simple_car_vendor_agent"

# simple prompt for car vendor agent who is friendly and helpful
CAR_VENDOR_PROMPT = "Your name is Suresh. You are a car vendor. You are friendly and helpful. You are curious and friendly, and have a sense of humor. your job is to help the client find the right car and then share screen and play video of the car. Also you should be asking if you want a video tour of the same."


class SimpleCarVendorAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=CAR_VENDOR_PROMPT)

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Hey I'm Suresh, your car vendor. How can I help you today?"
        )


def prewarm(proc: JobProcess):
    logger.info("Prewarming agent...")
    proc.userdata["vad"] = silero.VAD.load()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.close()
    logger.info("Prewarming agent complete")


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(model="nova-3"),
        tts=openai.TTS(voice="alloy"),
        turn_detection=EnglishModel(),
    )
    await ctx.wait_for_participant()
    agent = SimpleCarVendorAgent()
    agent.room = ctx.room
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


async def request_fnc(req: JobRequest):
    await req.accept(
        name=agent_display_name,
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            request_fnc=request_fnc,
            agent_name="simple_car_vendor_agent",
        )
    )
