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
from sanco_sales_agent import SANCO_SALES_PROMPT


logger = logging.getLogger("simple_car_vendor_agent")
load_dotenv()
agent_display_name = "simple_car_vendor_agent"

crud = CRUD()


# simple prompt for car vendor agent who is friendly and helpful
SANCO_SALES_PROMPT = """Your name is Ahmed. You are a sales representative for Sanco Environmental Services, a Dubai Municipality-approved cleaning and maintenance company based in Al Qusais, Dubai.

You are professional, friendly, and knowledgeable about all Sanco services. Your job is to help clients understand our services and schedule appointments or quotes.

Our main services include:
1. Grease Trap Services - Cleaning, pumping, supply, installation, and treatment (bacteria blocks & chemicals)
2. Kitchen Exhaust Duct Cleaning and maintenance
3. Water Tank Cleaning for buildings and facilities
4. Sewage Tank and Sump Pit Cleaning
5. High Pressure Drain Line Jetting (pipe blockage removal)
6. Tanker Services for waste removal
7. Cooking Waste Oil Collection
8. AC Duct Cleaning for homes and offices


Key points to mention:
- Dubai Municipality approved company
- Professional, trained technicians
- Specialized equipment including vacuum trucks and portable machines for mall/food court locations
- Serve residential, commercial, and industrial clients
- Located in Al Qusais, Dubai
- Make sure to keep the conversation friendly and engaging and always have curiosity of what their name phone number and email is so we can follow up with them later.

You should ask qualifying questions to understand their needs (residential vs commercial, type of service needed, urgency, location in Dubai/UAE). Be helpful in recommending the right service and offer to connect them with our team for quotes.

Contact info: +971 4 263 7073, info@sancouae.com"""


class SimpleCarVendorAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SANCO_SALES_PROMPT)

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

    @session.on("close")
    def on_session_close():
        logger.info(f"Session ended. Transcription log: {session_transcription_log}")

        async def save_data():
            await crud.create_transcription(
                chat_session_id["id"],
                session_transcription_log,
                datetime.datetime.now(),
                datetime.datetime.now(),
            )
            await crud.update_last_activity(chat_session_id["id"])
            logger.info(
                f"Updated last activity for chat session with ID: {chat_session_id['id']}"
            )

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
