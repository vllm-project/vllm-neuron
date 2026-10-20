# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import itertools
import logging
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_CONNECTION_RETRIES = 3
# Connection-level failures: the request never reached the app, so retrying is
# safe (the prefill call is max_tokens=1 and idempotent).
_RETRYABLE_ERRORS = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.ConnectTimeout,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize client pools for prefiller and decoder services
    app.state.prefill_clients = []
    app.state.decode_clients = []

    # Create prefill clients
    for i, (host, port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f"http://{host}:{port}/v1"
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=prefiller_base_url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    # Create decode clients
    for i, (host, port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f"http://{host}:{port}/v1"
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=decoder_base_url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    # Initialize round-robin iterators
    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))

    print(
        f"Initialized {len(app.state.prefill_clients)} prefill clients "
        f"and {len(app.state.decode_clients)} decode clients."
    )

    yield

    # Shutdown: Close all clients
    for client_info in app.state.prefill_clients:
        await client_info["client"].aclose()

    for client_info in app.state.decode_clients:
        await client_info["client"].aclose()


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    # Always use 127.0.0.1 as localhost binds to IPv6 which is blocked on CI
    parser.add_argument("--host", type=str, default="127.0.0.1")

    # For prefiller instances
    parser.add_argument(
        "--prefiller-hosts",
        "--prefiller-host",
        type=str,
        nargs="+",
        default=["localhost"],
    )
    parser.add_argument(
        "--prefiller-ports", "--prefiller-port", type=int, nargs="+", default=[8100]
    )

    # For decoder instances
    parser.add_argument(
        "--decoder-hosts", "--decoder-host", type=str, nargs="+", default=["localhost"]
    )
    parser.add_argument(
        "--decoder-ports", "--decoder-port", type=int, nargs="+", default=[8200]
    )

    args = parser.parse_args()

    # Extend hosts to match ports length (repeat last host if fewer hosts).
    if len(args.prefiller_hosts) < len(args.prefiller_ports):
        last = args.prefiller_hosts[-1] if args.prefiller_hosts else "localhost"
        args.prefiller_hosts.extend(
            [last] * (len(args.prefiller_ports) - len(args.prefiller_hosts))
        )
    if len(args.decoder_hosts) < len(args.decoder_ports):
        last = args.decoder_hosts[-1] if args.decoder_hosts else "localhost"
        args.decoder_hosts.extend(
            [last] * (len(args.decoder_ports) - len(args.decoder_hosts))
        )

    # Create tuples of (host, port) for each service type
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))

    return args


def get_next_client(app, service_type: str):
    """
    Get the next client in round-robin fashion.

    Args:
        app: The FastAPI app instance
        service_type: Either 'prefill' or 'decode'

    Returns:
        The next client to use
    """
    if service_type == "prefill":
        client_idx = next(app.state.prefill_iterator)
        return app.state.prefill_clients[client_idx]
    elif service_type == "decode":
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")


async def _post_with_retry(
    client: httpx.AsyncClient, endpoint: str, req_data: dict, headers: dict
):
    """POST to a backend, retrying connection-level failures.

    Under load the backend's keep-alive timeout can close a pooled connection
    while the pool still believes it is usable; the next send then fails with
    httpx.ReadError before reaching the app. httpx discards the dead connection
    on failure, so a retry dials a fresh one and succeeds.
    """
    last_exc = None
    for attempt in range(_CONNECTION_RETRIES):
        try:
            response = await client.post(endpoint, json=req_data, headers=headers)
            response.raise_for_status()
            return response
        except _RETRYABLE_ERRORS as e:
            last_exc = e
            logger.warning(
                "Connection error posting to %s (attempt %d/%d): %s: %s",
                endpoint,
                attempt + 1,
                _CONNECTION_RETRIES,
                type(e).__name__,
                e,
            )
    raise last_exc or RuntimeError(
        f"_post_with_retry made no attempt (_CONNECTION_RETRIES={_CONNECTION_RETRIES})"
    )


async def send_request_to_service(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """
    Send a request to a service using a client from the pool.
    """
    req_data = req_data.copy()
    # If the client already provided kv_transfer_params (multi-turn with
    # remote_block_ids from a previous D response), merge them in.
    # Otherwise, use the default P→D params.
    if "kv_transfer_params" not in req_data or not req_data.get("kv_transfer_params"):
        req_data["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    response = await _post_with_retry(
        client_info["client"], endpoint, req_data, headers
    )

    # Read/consume the response body to release the connection.
    # Without this, the connection stays open and causes http.ReadError
    # on subsequent requests due to connection pool exhaustion.
    await response.aread()

    return response


async def stream_service_response(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """
    Asynchronously stream response from a service using a client from the pool.

    Opening the stream reads the response headers, so it can hit the same stale
    keep-alive connection as a plain POST. Retries cover the open only: once a
    chunk has been yielded downstream the response cannot be replayed.
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    for attempt in range(_CONNECTION_RETRIES):
        yielded_any = False
        try:
            async with client_info["client"].stream(
                "POST", endpoint, json=req_data, headers=headers
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yielded_any = True
                    yield chunk
            return
        except _RETRYABLE_ERRORS as e:
            if yielded_any or attempt == _CONNECTION_RETRIES - 1:
                raise
            logger.warning(
                "Connection error opening stream to %s (attempt %d/%d): %s: %s",
                endpoint,
                attempt + 1,
                _CONNECTION_RETRIES,
                type(e).__name__,
                e,
            )


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())
        client_wants_stream = req_data.get("stream", False)

        # Preserve client-provided kv_transfer_params (multi-turn D→P).
        client_kv_params = req_data.pop("kv_transfer_params", None)

        # Get the next prefill client in round-robin fashion
        prefill_client_info = get_next_client(request.app, "prefill")

        # Build prefill request data. If client sent kv_transfer_params
        # with remote_block_ids (from a previous D response), forward them
        # so P can pull KV from D.
        prefill_req_data = req_data.copy()
        if client_kv_params and client_kv_params.get("remote_block_ids"):
            prefill_req_data["kv_transfer_params"] = client_kv_params

        # Send request to prefill service (always non-streaming, max_tokens=1)
        response = await send_request_to_service(
            prefill_client_info, api, prefill_req_data, request_id
        )

        # Extract kv_transfer_params from prefill response
        response_json = response.json()
        await response.aclose()
        kv_transfer_params = response_json.get("kv_transfer_params", {})
        if kv_transfer_params:
            kv_transfer_params["remote_host"] = prefill_client_info["host"]
            req_data["kv_transfer_params"] = kv_transfer_params

        # Get the next decode client in round-robin fashion
        decode_client_info = get_next_client(request.app, "decode")

        logger.debug("Using %s %s", prefill_client_info, decode_client_info)

        if client_wants_stream:
            # Streaming: pass through the decode SSE stream to the client.
            # The vllm server includes kv_transfer_params in the final chunk
            # (via our streaming patch). We need to fix remote_host in that
            # chunk since the D-node reports 0.0.0.0 as its listen address.
            decode_req_data = req_data.copy()
            decode_req_data["stream"] = True
            decode_host = decode_client_info["host"]

            async def generate_stream():
                import json as _json

                async for chunk in stream_service_response(
                    decode_client_info,
                    api,
                    decode_req_data,
                    request_id=request_id,
                ):
                    # Fix remote_host in kv_transfer_params if present
                    text = (
                        chunk.decode("utf-8", errors="replace")
                        if isinstance(chunk, bytes)
                        else chunk
                    )
                    if "kv_transfer_params" in text and "remote_host" in text:
                        try:
                            for line in text.split("\n"):
                                if (
                                    line.startswith("data: ")
                                    and "kv_transfer_params" in line
                                ):
                                    data = _json.loads(line[6:])
                                    if data.get("kv_transfer_params", {}).get(
                                        "remote_host"
                                    ):
                                        data["kv_transfer_params"]["remote_host"] = (
                                            decode_host
                                        )
                                        fixed = f"data: {_json.dumps(data)}\n"
                                        chunk = (
                                            fixed.encode()
                                            if isinstance(chunk, bytes)
                                            else fixed
                                        )
                        except (ValueError, KeyError):
                            pass
                    yield chunk

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
            )
        else:
            # Non-streaming: buffer decode response to capture kv_transfer_params.
            decode_req_data = req_data.copy()
            decode_req_data["stream"] = False
            if "stream_options" in decode_req_data:
                del decode_req_data["stream_options"]
            headers = {
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
                "X-Request-Id": request_id,
            }
            decode_response = await _post_with_retry(
                decode_client_info["client"], api, decode_req_data, headers
            )
            await decode_response.aread()
            decode_json = decode_response.json()
            await decode_response.aclose()

            d_kv_params = decode_json.get("kv_transfer_params")
            if d_kv_params:
                d_kv_params["remote_host"] = decode_client_info["host"]
                decode_json["kv_transfer_params"] = d_kv_params

            from fastapi.responses import JSONResponse

            return JSONResponse(content=decode_json)

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("chat/completions", request)


@app.post("/start_profile")
async def start_profile():
    """Fan out /start_profile to all backend servers in parallel."""
    import asyncio

    all_clients = app.state.prefill_clients + app.state.decode_clients

    async def _start_one(client_info):
        base_url = str(client_info["client"].base_url)
        # /start_profile is on the root, not under /v1
        profile_url = base_url.rsplit("/v1", 1)[0] + "/start_profile"
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(profile_url)
            resp.raise_for_status()
            return resp.json()

    results = await asyncio.gather(
        *[_start_one(c) for c in all_clients], return_exceptions=True
    )
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        logger.error("start_profile failed on some backends: %s", errors)
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=500,
            content={"error": str(errors[0]), "failed_count": len(errors)},
        )
    return {"status": "ok", "backends_profiled": len(all_clients)}


@app.post("/stop_profile")
async def stop_profile():
    """Fan out /stop_profile to all backend servers in parallel."""
    import asyncio

    all_clients = app.state.prefill_clients + app.state.decode_clients

    async def _stop_one(client_info):
        base_url = str(client_info["client"].base_url)
        profile_url = base_url.rsplit("/v1", 1)[0] + "/stop_profile"
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(profile_url)
            resp.raise_for_status()
            return resp.json()

    results = await asyncio.gather(
        *[_stop_one(c) for c in all_clients], return_exceptions=True
    )
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        logger.error("stop_profile failed on some backends: %s", errors)
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=500,
            content={"error": str(errors[0]), "failed_count": len(errors)},
        )
    return {"status": "ok", "backends_stopped": len(all_clients)}


@app.get("/healthcheck")
async def healthcheck():
    """Simple endpoint to check if the server is running."""
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
    }


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
