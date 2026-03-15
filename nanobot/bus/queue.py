"""Async message queue for decoupled channel-agent communication."""

import asyncio

from nanobot.bus.events import InboundMessage, OutboundMessage


# Maximum number of outbound messages allowed in the queue.
# Progress/tool-hint messages are dropped when the queue is full to prevent
# memory growth from bursts. Regular (non-progress) messages always wait.
_OUTBOUND_MAXSIZE = 200


class MessageBus:
    """
    Async message bus that decouples chat channels from the agent core.

    Channels push messages to the inbound queue, and the agent processes
    them and pushes responses to the outbound queue.

    The outbound queue is bounded (maxsize=200). Progress/tool-hint messages
    are silently dropped when the queue is full; regular messages block until
    space is available to ensure delivery.
    """

    def __init__(self):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue(maxsize=_OUTBOUND_MAXSIZE)

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish a message from a channel to the agent."""
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """Consume the next inbound message (blocks until available)."""
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Publish a response from the agent to channels.

        Progress/tool-hint messages are dropped when the queue is full.
        Regular messages block until space is available.
        """
        is_progress = bool(msg.metadata.get("_progress"))
        if is_progress:
            try:
                self.outbound.put_nowait(msg)
            except asyncio.QueueFull:
                pass  # silently drop progress updates under backpressure
        else:
            await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        """Consume the next outbound message (blocks until available)."""
        return await self.outbound.get()

    @property
    def inbound_size(self) -> int:
        """Number of pending inbound messages."""
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """Number of pending outbound messages."""
        return self.outbound.qsize()
