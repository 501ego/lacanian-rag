import asyncio
import logging
import types

from app.interfaces import logger as logger_module


def test_request_id_filter_sets_fields():
    record = logging.LogRecord("name", logging.INFO, "", 0, "msg", (), None)
    filt = logger_module.RequestIdFilter()
    assert filt.filter(record) is True
    assert hasattr(record, "request_id")
    assert hasattr(record, "trace_id")


def test_color_formatter():
    formatter = logger_module.ColorFormatter("%(levelname)s", use_color=False)
    record = logging.LogRecord("name", logging.INFO, "", 0, "msg", (), None)
    assert formatter.format(record) == "INFO"


def test_configure_logger_reuses():
    logger = logger_module.configure_logger("test_logger")
    logger_again = logger_module.configure_logger("test_logger")
    assert logger is logger_again


def test_request_logging_middleware():
    logger = logger_module.configure_logger("middleware_logger")
    middleware = logger_module.request_logging_middleware(logger)

    class DummyURL:
        def __init__(self):
            self.path = "/path"
            self.query = "a=1"

    class DummyClient:
        host = "127.0.0.1"

    class DummyRequest:
        def __init__(self):
            self.headers = {}
            self.method = "GET"
            self.url = DummyURL()
            self.client = DummyClient()

    async def call_next(_request):
        return types.SimpleNamespace(status_code=200, headers={})

    async def run():
        response = await middleware(DummyRequest(), call_next)
        assert response.headers.get("X-Request-Id")

    asyncio.run(run())
