import logging
import asyncio
import datetime  # Add this import
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
from livekit.plugins import silero, deepgram, openai, tavus
from livekit.plugins.turn_detector.english import EnglishModel
from db.crud import CRUD


logger = logging.getLogger("simple_car_vendor_agent")
load_dotenv()
agent_display_name = "simple_car_vendor_agent"

crud = CRUD()


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
    session_transcription_log = ""
    chat_session_id = await crud.create_chat_session(session_type="VIDEO")
    logger.info(f"Created chat session with ID: {chat_session_id['id']}")

    avatar = tavus.AvatarSession(
        replica_id=os.getenv("TAVUS_REPLICA_ID"),
        persona_id=os.getenv("TAVUS_PERSONA_ID"),
    )

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(model="nova-3"),
        tts=openai.TTS(voice="alloy"),
        turn_detection=EnglishModel(),
    )

    # Move the event handler inside entrypoint where session is defined
    @session.on("user_input_transcribed")
    def on_transcript(transcript):
        nonlocal session_transcription_log
        if transcript.is_final:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"[{timestamp}] User: {transcript.transcript}")
            session_transcription_log += (
                f"[{timestamp}] User: {transcript.transcript}\n"
            )

    # Optional: Log agent responses too
    @session.on("conversation_item_added")
    def on_conversation_item_added(event):
        nonlocal session_transcription_log
        if event.item.type == "message" and event.item.role == "assistant":
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            text = event.item.text_content
            if text:
                logger.info(f"[{timestamp}] Agent: {text}")
                session_transcription_log += f"[{timestamp}] Agent: {text}\n"

    await ctx.wait_for_participant()
    agent = SimpleCarVendorAgent()
    agent.room = ctx.room

    await avatar.start(session, room=ctx.room)

    result = await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(
            transcription_enabled=True,
            audio_enabled=False,
        ),
    )

    if avatar.conversation_id:
        logger.info(f"Tavus Conversation ID: {avatar.conversation_id}")
        logger.info(
            f"To get the report/transcript, use: GET https://tavusapi.com/v2/conversations/{avatar.conversation_id}?verbose=true"
        )

    @session.on('close')
    def on_session_close():
        logger.info(f"Session ended. Transcription log: {session_transcription_log}")
        
        async def save_data():
            await crud.create_transcription(chat_session_id['id'], session_transcription_log, datetime.datetime.now(), datetime.datetime.now())
            await crud.update_last_activity(chat_session_id['id'])
            logger.info(f"Updated last activity for chat session with ID: {chat_session_id['id']}")

        asyncio.create_task(save_data())

    if result:
        await result.done()

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
