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


logger = logging.getLogger("sanco_sales_agent")
load_dotenv()
agent_display_name = "sanco_sales_agent"


# Prompt for Sanco Environmental Services sales agent
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

You should ask qualifying questions to understand their needs (residential vs commercial, type of service needed, urgency, location in Dubai/UAE). Be helpful in recommending the right service and offer to connect them with our team for quotes.

Contact info: +971 4 263 7073, info@sancouae.com"""



class SancoSalesAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SANCO_SALES_PROMPT)


    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Greet the customer warmly. Introduce yourself as Ahmed from Sanco Environmental Services and ask how you can help them today."
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
        tts=openai.TTS(voice="echo"),  # Changed to 'echo' for more professional male voice
        turn_detection=EnglishModel(),
    )
    await ctx.wait_for_participant()
    agent = SancoSalesAgent()
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
            agent_name="sanco_sales_agent",
        )
    )
