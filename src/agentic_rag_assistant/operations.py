"""Bound request bodies and emit useful diagnostics without logging user content."""

import json
import logging
from time import perf_counter
from uuid import uuid4

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from agentic_rag_assistant.ingestion import MAX_DOCUMENT_BYTES

# Leave room for a single upload's multipart envelope, not a second large file.
MAX_REQUEST_BYTES = MAX_DOCUMENT_BYTES + 64 * 1024
LOGGER = logging.getLogger("agentic_rag_assistant.requests")


def configure_request_logging() -> None:
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default response includes rejected inputs; questions may be private.
    errors = [
        {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


class RequestDiagnosticsMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = perf_counter()
        identifier = uuid4().hex
        scope.setdefault("state", {})["request_id"] = identifier
        status, sent, consumed, error_type = 500, False, 0, None

        async def bounded_receive() -> Message:
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > MAX_REQUEST_BYTES:
                    raise HTTPException(413, "The request body exceeds the upload limit.")
            return message

        async def observed_send(message: Message) -> None:
            nonlocal status, sent
            if message["type"] == "http.response.start":
                status, sent = message["status"], True
                MutableHeaders(scope=message)["X-Request-ID"] = identifier
            await send(message)

        try:
            lengths = [v for k, v in scope.get("headers", []) if k.lower() == b"content-length"]
            if lengths:
                try:
                    if len(lengths) != 1 or not lengths[0].isdigit():
                        raise ValueError
                    length = int(lengths[0])
                except ValueError:
                    await JSONResponse(status_code=400, content={
                        "detail": "Invalid Content-Length header.",
                    })(scope, receive, observed_send)
                    return
                if length > MAX_REQUEST_BYTES:
                    await JSONResponse(status_code=413, content={
                        "detail": "The request body exceeds the upload limit.",
                    })(scope, receive, observed_send)
                    return
            await self.app(scope, bounded_receive, observed_send)
        except Exception as exc:
            error_type = type(exc).__name__
            if sent:
                # Never forward an exception whose message may contain credentials.
                raise RuntimeError("The response failed after it started.") from None
            await JSONResponse(status_code=500, content={
                "detail": "An unexpected server error occurred. Use the request ID for support.",
            })(scope, receive, observed_send)
        finally:
            route = scope.get("route")
            method = scope.get("method", "OTHER")
            entry = {
                "event": "http_request", "request_id": identifier,
                "method": method if method in {
                    "GET", "HEAD", "POST", "PATCH", "PUT", "DELETE", "OPTIONS",
                } else "OTHER",
                "route": getattr(route, "path", "<unmatched>"), "status": status,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
            }
            if error_type:
                entry["error_type"] = error_type
            LOGGER.info(json.dumps(entry, separators=(",", ":")))
