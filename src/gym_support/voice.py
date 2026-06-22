"""Pipecat voice bot for the gym support graph.

Wraps the same multi-agent LangGraph from ``graph.py`` in a real-time voice
pipeline:

    transport in → STT → context (user) → LLM → TTS → transport out → context (assistant)

The ``LLM`` stage is ``LangGraphLLMService``: it runs our whole graph as the
brain, so going voice doesn't change the agent — it wraps it. The rest of the
pipeline (transport, STT, TTS, Silero VAD, context aggregators) is stock Pipecat.

Also wired up:
  * **LangSmith tracing** — Pipecat's OTel spans (turn/stt/llm/tts) bridged to
    LangSmith, with the graph's nodes nested under the ``llm`` span.
  * **Conversation recording** — the whole session captured as a stereo WAV
    (user left / bot right) and attached to the LangSmith root span.

Run it (opens a browser client via Pipecat's dev runner):

    uv run python -m gym_support.voice

Needs ``OPENAI_API_KEY`` (STT + TTS) in ``.env``, plus the agent's model key
(``ANTHROPIC_API_KEY`` by default) and the LangSmith / OTEL vars for tracing.
"""

from __future__ import annotations

import os
import tempfile
import uuid
import wave

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

from .langgraph_llm_service import LangGraphLLMService
from .graph import build_graph
from .processor import setup_langsmith_tracing

load_dotenv(override=True)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    """Build and run the voice pipeline over the given transport."""

    conversation_id = str(uuid.uuid4())

    tracing_processor = setup_langsmith_tracing(
        llm_span_kind="chain",
        thread_id_provider=lambda: conversation_id,
    )

    # Record the whole conversation as a stereo WAV (user left / bot right) and
    # register the path so the LangSmith root span gets the audio attached.
    recording_path = os.path.join(tempfile.gettempdir(), f"langgym-{conversation_id}.wav")
    audiobuffer = AudioBufferProcessor(num_channels=2)
    tracing_processor.register_recording(conversation_id, recording_path)

    # --- Speech in -----------------------------------------------------------
    stt = OpenAISTTService(api_key=os.getenv("OPENAI_API_KEY"))

    # --- LLM: our LangGraph runs as the brain (see langgraph_llm_service.py) -
    llm = LangGraphLLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        graph=build_graph(),
    )


    # --- Speech out ----------------------------------------------------------
    tts = OpenAITTSService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAITTSService.Settings(voice="alloy"),
    )

    # The context is the conversation's single source of truth. The user
    # aggregator also carries Silero VAD, which delimits turns and enables
    # barge-in (interrupting the bot mid-sentence).
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),   # mic in
            stt,                 # speech → text
            user_aggregator,     # append the user turn to the context
            llm,                 # the LangGraph brain: text → reply (+ tool calls)
            tts,                 # text → speech
            transport.output(),  # speaker out
            audiobuffer,         # tap user + bot audio for the recording
            assistant_aggregator,  # append the spoken reply to the context
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        enable_tracing=True,             # emit conversation/turn/stt/llm/tts spans
        conversation_id=conversation_id,  # root span id == the LangSmith thread
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):  # noqa: ANN001
        logger.info("Client connected")
        await audiobuffer.start_recording()
        # Speak a fixed greeting (no LLM call) and record it as the assistant's
        # first turn, so the first real user message has something to follow.
        await worker.queue_frames(
            [
                TTSSpeakFrame(
                    "Hi! Welcome to LangGym support. How can I help you today?",
                    append_to_context=True,
                )
            ]
        )

    @audiobuffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels):  # noqa: ANN001
        # Fires once on stop_recording(): write the merged stereo WAV to the path
        # the tracing processor reads and attaches to the conversation span.
        with wave.open(recording_path, "wb") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(2)  # PCM16
            wf.setframerate(sample_rate)
            wf.writeframes(audio)
        logger.info(f"Saved conversation recording: {recording_path}")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):  # noqa: ANN001
        logger.info("Client disconnected")
        # Stop first so the WAV is written before the conversation span ends and
        # the tracing processor reads it.
        await audiobuffer.stop_recording()
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments) -> None:
    """Entry point the Pipecat runner calls; picks a transport and runs the bot."""
    transport_params = {
        "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    }
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
