import os
import sys
import types
import importlib.util


def _install_module(name, module):
    sys.modules[name] = module


def _module_missing(name):
    return importlib.util.find_spec(name) is None


os.environ.setdefault("OPENAI_API_KEY", "test-key")


# Stub openai when missing to avoid hard dependency.
if _module_missing("openai"):
    openai_stub = types.ModuleType("openai")

    class OpenAIError(Exception):
        pass

    class _DummyChat:
        def __init__(self):
            self.completions = self

        def create(self, **_kwargs):
            return types.SimpleNamespace(choices=[])

    class _DummyEmbeddings:
        def create(self, **_kwargs):
            return types.SimpleNamespace(data=[])

    class OpenAI:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.chat = _DummyChat()
            self.embeddings = _DummyEmbeddings()

    openai_stub.OpenAI = OpenAI
    openai_stub.OpenAIError = OpenAIError
    _install_module("openai", openai_stub)


# Stub dotenv when missing.
if _module_missing("dotenv"):
    dotenv_stub = types.ModuleType("dotenv")

    def load_dotenv(*_args, **_kwargs):
        return None

    dotenv_stub.load_dotenv = load_dotenv
    _install_module("dotenv", dotenv_stub)


# Stub faiss when missing.
if _module_missing("faiss"):
    faiss_stub = types.ModuleType("faiss")

    class IndexFlatL2:
        def __init__(self, d):
            self.d = d
            self._data = []

        def add(self, vectors):
            self._data.append(vectors)

        def search(self, vector, top_k):
            distances = [[0.0 for _ in range(top_k)]]
            indices = [[0 for _ in range(top_k)]]
            return distances, indices

    def read_index(_path):
        return IndexFlatL2(1)

    def write_index(_index, _path):
        return None

    faiss_stub.IndexFlatL2 = IndexFlatL2
    faiss_stub.read_index = read_index
    faiss_stub.write_index = write_index
    _install_module("faiss", faiss_stub)


# Stub numpy when missing.
if _module_missing("numpy"):
    numpy_stub = types.ModuleType("numpy")

    class _FakeArray:
        def __init__(self, data):
            self.data = data
            if isinstance(data, list) and data and isinstance(data[0], list):
                self.shape = (len(data), len(data[0]))
                self.ndim = 2
            else:
                length = len(data) if isinstance(data, list) else 1
                self.shape = (length,)
                self.ndim = 1

        def reshape(self, rows, cols):
            if cols == -1:
                cols = self.shape[0]
            self.shape = (rows, cols)
            self.ndim = 2
            return self

    def array(data, dtype=None):
        _ = dtype
        return _FakeArray(data)

    numpy_stub.array = array
    numpy_stub.ndarray = _FakeArray
    _install_module("numpy", numpy_stub)


# Stub PyPDF2 when missing.
if _module_missing("PyPDF2"):
    pypdf2_stub = types.ModuleType("PyPDF2")

    class PdfReader:
        def __init__(self, _path):
            self.pages = []

    pypdf2_stub.PdfReader = PdfReader
    _install_module("PyPDF2", pypdf2_stub)


# Stub tqdm when missing.
if _module_missing("tqdm"):
    tqdm_stub = types.ModuleType("tqdm")

    def tqdm(iterable, **_kwargs):
        return iterable

    tqdm_stub.tqdm = tqdm
    _install_module("tqdm", tqdm_stub)


# Stub fastapi/pydantic when missing.
if _module_missing("fastapi"):
    fastapi_stub = types.ModuleType("fastapi")
    responses_stub = types.ModuleType("fastapi.responses")

    class HTTPException(Exception):
        def __init__(self, status_code=500, detail=None):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FastAPI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def post(self, *_args, **_kwargs):
            def decorator(func):
                return func
            return decorator

        def middleware(self, *_args, **_kwargs):
            def decorator(func):
                return func
            return decorator

    def File(default=None, **_kwargs):
        return default

    def Form(default=None, **_kwargs):
        return default

    class UploadFile:
        def __init__(self, filename=""):
            self.filename = filename

    class Request:
        def __init__(self, method="GET", url=None, headers=None, client=None):
            self.method = method
            self.url = url
            self.headers = headers or {}
            self.client = client

    class Response:
        def __init__(self, status_code=200):
            self.status_code = status_code
            self.headers = {}

    class StreamingResponse:
        def __init__(self, content, media_type=None, headers=None):
            self.content = content
            self.media_type = media_type
            self.headers = headers or {}

    fastapi_stub.FastAPI = FastAPI
    fastapi_stub.File = File
    fastapi_stub.Form = Form
    fastapi_stub.HTTPException = HTTPException
    fastapi_stub.UploadFile = UploadFile
    fastapi_stub.Request = Request
    fastapi_stub.Response = Response

    responses_stub.StreamingResponse = StreamingResponse

    _install_module("fastapi", fastapi_stub)
    _install_module("fastapi.responses", responses_stub)

if _module_missing("pydantic"):
    pydantic_stub = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)

        def dict(self):
            return dict(self.__dict__)

        def model_dump(self):
            return dict(self.__dict__)

    def Field(default=None, **_kwargs):
        return default

    pydantic_stub.BaseModel = BaseModel
    pydantic_stub.Field = Field
    _install_module("pydantic", pydantic_stub)
