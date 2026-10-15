"""Loopback-only workbench for verified, curated development tasks."""
import argparse
import asyncio
import hashlib
import json
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .data import load_task, verify
from .jobs import BusyError, JobStore
from .regressions import CASES


class RepairRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    task: str = Field(pattern='^[a-z_]+$', max_length=40)
    budget: int = Field(default=4, ge=1, le=4)
    feedback: bool = True
    seconds: int = Field(default=300, ge=10, le=600)


class LocalBoundary:
    def __init__(self, app, token, port):
        self.app, self.token = app, token
        self.hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
        self.origins = {'http://' + host for host in self.hosts}

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = dict(scope['headers'])
        host = headers.get(b'host', b'').decode('latin1')
        origin = headers.get(b'origin', b'').decode('latin1')
        error = None
        if host not in self.hosts or (origin and origin not in self.origins):
            error = (403, 'Only the local workbench origin is allowed')
        elif scope['method'] not in {'GET', 'HEAD'} and not secrets.compare_digest(headers.get(b'x-repair-token', b''), self.token.encode()):
            error = (403, 'Missing or invalid workbench token')
        if error:
            return await JSONResponse({'detail': error[1]}, error[0])(scope, receive, send)
        # Buffer only a tiny bounded JSON body, including requests without Content-Length.
        messages, size = [], 0
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            size += len(message.get('body', b''))
            if size > 1024:
                return await JSONResponse({'detail': 'Request body exceeds 1024 bytes'}, 413)(scope, receive, send)
            messages.append(message)
            if not message.get('more_body', False):
                break
        async def replay():
            return messages.pop(0) if messages else await receive()
        async def secure_send(message):
            if message['type'] == 'http.response.start':
                message['headers'] += [(b'cache-control', b'no-store'), (b'x-content-type-options', b'nosniff'),
                    (b'content-security-policy', b"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")]
            await send(message)
        await self.app(scope, replay, secure_send)


def create_app(store, *, port=8018, assets=None):
    token = secrets.token_urlsafe(32)
    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await asyncio.to_thread(store.close)
    app = FastAPI(title='Repair workbench', lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(LocalBoundary, token=token, port=port)

    @app.get('/api/config')
    def config():
        return {'token': token, 'tasks': sorted(CASES), 'identity': store.identity,
                'limits': {'concurrency': 1, 'jobs': store.capacity, 'attempts': 4, 'seconds': 600,
                           'request_bytes': 1024, 'output_tokens': 768},
                'notice': 'Public development examples. Passing tests does not prove correctness.'}

    @app.get('/api/health')
    def health():
        return {'ready': not store.closed, 'active_job': store.active}

    @app.get('/api/tasks/{name}')
    def task(name: str):
        if name not in CASES:
            raise HTTPException(404, 'Unknown curated task')
        item = load_task(store.root, name)
        return {'name': name, 'source': item.source}

    @app.get('/api/jobs')
    def jobs():
        return store.list()

    @app.post('/api/jobs', status_code=202)
    def submit(request: RepairRequest):
        try:
            return store.submit(request.task, request.budget, request.feedback, request.seconds)
        except BusyError as error:
            raise HTTPException(429, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get('/api/jobs/{job_id}')
    def job(job_id: str):
        try:
            return store.get(job_id)
        except KeyError as error:
            raise HTTPException(404, 'Unknown job') from error

    @app.post('/api/jobs/{job_id}/cancel')
    def cancel(job_id: str):
        try:
            return store.cancel(job_id)
        except KeyError as error:
            raise HTTPException(404, 'Unknown job') from error

    @app.get('/api/jobs/{job_id}/export')
    def export(job_id: str):
        item = job(job_id)
        return JSONResponse(item, headers={'Content-Disposition': f'attachment; filename="repair-{item["id"]}.json"'})

    assets = Path(assets) if assets else Path(__file__).parent / 'web'
    if assets.is_dir():
        app.mount('/', StaticFiles(directory=assets, html=True), name='workbench')
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/quixbugs'))
    parser.add_argument('--models', type=Path, default=Path('data/models'))
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--image-file', type=Path, default=Path('runs/image.id'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--port', type=int, default=8018)
    parser.add_argument('--session-seconds', type=int, default=3600)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or not 60 <= args.session_seconds <= 21600:
        parser.error('Use port 1024..65535 and session duration 60..21600 seconds')
    import uvicorn
    from .inference import CodeModel
    from .sandbox import DockerRunner
    manifest = verify(args.data)
    runner = DockerRunner(args.image_file.read_text().strip())
    runtime = runner.preflight()
    generator = CodeModel(args.models, args.device, 768, adapter=args.adapter)
    identity = {'dataset_revision': manifest['revision'], 'runtime': runtime,
                'model': generator.record, 'adapter': generator.adapter_record,
                'implementation_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path(__file__).parent.glob('*.py'))}}
    store = JobStore(args.data, args.output, runner, generator, identity)
    app = create_app(store, port=args.port)
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=args.port, workers=1,
        proxy_headers=False, access_log=False, limit_concurrency=16, timeout_keep_alive=5))
    async def serve():
        async def expire():
            await asyncio.sleep(args.session_seconds)
            server.should_exit = True
        timer = asyncio.create_task(expire())
        try:
            await server.serve()
        finally:
            timer.cancel()
            await asyncio.to_thread(store.close)
    asyncio.run(serve())


if __name__ == '__main__':
    main()
