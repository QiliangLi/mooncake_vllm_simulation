import ctypes as C
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class Conductor:
    """Calls the original Mooncake C++ PrefixCacheTable; this is not a router."""
    def __init__(self, block_size, workers):
        self.block_size = block_size
        self.lib = C.CDLL(str(ROOT / 'build/libconductor_lab.so'))
        signatures = {
            'mc_new': ([C.c_int], C.c_void_p),
            'mc_delete': ([C.c_void_p], None),
            'mc_error': ([], C.c_char_p),
            'mc_register': ([C.c_void_p, C.c_char_p], C.c_int),
            'mc_seed': ([C.c_void_p, C.POINTER(C.c_int32), C.c_int, C.c_char_p, C.c_int], C.c_int),
            'mc_query': ([C.c_void_p, C.POINTER(C.c_int32), C.c_int, C.c_char_p], C.c_int64),
            'mc_hash': ([C.c_void_p, C.POINTER(C.c_int32), C.c_int, C.POINTER(C.c_uint64)], C.c_int),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, name); fn.argtypes = args; fn.restype = result
        self.ptr = self.lib.mc_new(block_size)
        if not self.ptr: raise RuntimeError(self.lib.mc_error().decode())
        for w in workers: self.check(self.lib.mc_register(self.ptr, w.encode()))

    def check(self, result):
        if result < 0: raise RuntimeError(self.lib.mc_error().decode())
        return result

    @staticmethod
    def array(tokens):
        if any(not 0 <= t < 2**31 for t in tokens): raise ValueError('token id outside int32 range')
        return (C.c_int32 * len(tokens))(*tokens)

    def seed(self, tokens, object_id, remove=False):
        self.check(self.lib.mc_seed(self.ptr, self.array(tokens), len(tokens), object_id.encode(), remove))

    def query(self, tokens, worker):
        return self.lib.mc_query(self.ptr, self.array(tokens), len(tokens), worker.encode())

    def hashes(self, tokens):
        out = (C.c_uint64 * (len(tokens) // self.block_size))()
        n = self.check(self.lib.mc_hash(self.ptr, self.array(tokens), len(tokens), out))
        return list(out[:n])

    def close(self):
        if self.ptr: self.lib.mc_delete(self.ptr); self.ptr = None
