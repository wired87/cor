SHIFT_DIRS = [
     [
        [ 1, 0, 0], [0,  1, 0], [0, 0,  1],
        [ 1, 1, 0], [1, 0,  1], [0, 1,  1],
        [ 1,-1, 0], [1, 0, -1], [0, 1, -1],
        [ 1, 1, 1], [1, 1,-1], [1,-1, 1], [1,-1,-1],
     ],
     [
        [-1, 0, 0], [0, -1, 0], [0, 0, -1],
        [-1,-1, 0], [-1, 0,-1], [0,-1,-1],
        [-1, 1, 0], [-1, 0, 1], [0,-1, 1],
        [-1,-1,-1], [-1,-1, 1], [-1, 1,-1], [-1, 1, 1],
    ]
]

DIM = 3

import ast
import re
import jax.numpy as jnp
import jax

# CHAR: exec'd equation code calls `jnp.matmul` / `jnp.dot` directly. Composed
# expressions can yield 0-D scalars (after sum/mean) as operands, which JAX
# rejects with "ndim at least 1". We expose a thin `jnp` shim that auto-promotes
# scalars to 1-element vectors for matmul / dot only — every other attribute is
# transparently delegated to the real `jax.numpy` so semantics stay unchanged.
class _SafeJnp:
    __slots__ = ("_jnp",)

    def __init__(self, _jnp_module):
        object.__setattr__(self, "_jnp", _jnp_module)

    @staticmethod
    def _safe_pair(x, p):
        # CHAR: scalar * array is the natural fallback when either operand collapses
        # to 0-D — preserves the multiplicative meaning of matmul / dot while staying
        # broadcast-safe; ndim>=1 operands flow through unchanged for true matmul/dot.
        x = jnp.asarray(x)
        p = jnp.asarray(p)
        return x, p

    def matmul(self, x, p):
        x, p = self._safe_pair(x, p)
        if x.ndim == 0 or p.ndim == 0:
            return x * p
        return self._jnp.matmul(x, p)

    def dot(self, x, p):
        x, p = self._safe_pair(x, p)
        if x.ndim == 0 or p.ndim == 0:
            return x * p
        return self._jnp.dot(x, p)

    def __getattr__(self, name):
        return getattr(self._jnp, name)


LIBS={
    "jax": jax,
    "vmap": jax.vmap,
    "jnp": _SafeJnp(jnp),
    "jit": jax.jit,
}


class _MatMulRewriter(ast.NodeTransformer):
    # CHAR: rewrite `a @ b` (ast.MatMult) → `jnp.matmul(a, b)` so the SafeJnp shim
    # above can transparently fall back to scalar broadcasting when an operand
    # collapses to 0-D inside an equation chain (e.g. `a @ b @ c` after `a @ b`
    # already returned a scalar). The transformation preserves left-associativity.
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.MatMult):
            call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="jnp", ctx=ast.Load()),
                    attr="matmul",
                    ctx=ast.Load(),
                ),
                args=[node.left, node.right],
                keywords=[],
            )
            return ast.copy_location(call, node)
        return node


def _rewrite_matmul_operator(code: str) -> str:
    """Replace every `@` matmul-operator usage with `jnp.matmul(...)` so the SafeJnp
    shim governs the call. Decorators (`@deco`) are left untouched because they are
    parsed as `ast.FunctionDef.decorator_list`, not `BinOp(MatMult)`."""
    try:
        tree = ast.parse(code)
        tree = _MatMulRewriter().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception as exc:
        print(f"Warn _rewrite_matmul_operator: {type(exc).__name__}: {exc}")
        return code


def _fix_einsum_unquoted_subscript(code: str) -> str:
    """
    CHAR: Some graphs store e.g. `jnp.einsum(i,ij->j, a, b)` (quotes dropped). Repair to
    `jnp.einsum("i,ij->j", a, b)` so `exec` succeeds.
    """
    return re.sub(
        r'jnp\.einsum\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*([a-zA-Z0-9,]+->[a-zA-Z0-9,]+)\s*,',
        r'jnp.einsum("\1,\2",',
        code,
    )


def create_runnable(eq_code):
    try:
        if isinstance(eq_code, dict):
            eq_code = (
                eq_code.get("code")
                or eq_code.get("src")
                or eq_code.get("body")
                or ""
            )
        if eq_code is None or (isinstance(eq_code, str) and not str(eq_code).strip()):

            def _noop(*_a, **_k):
                return jnp.asarray(0.0, dtype=jnp.float32)

            return _noop
        if not isinstance(eq_code, (str, bytes)):
            print(f"Warn create_runnable: need str code, got {type(eq_code).__name__}")

            def _noop(*_a, **_k):
                return jnp.asarray(0.0, dtype=jnp.float32)

            return _noop

        namespace = {}
        if isinstance(eq_code, str):
            eq_code = _fix_einsum_unquoted_subscript(eq_code)
            # CHAR: route every `@` matmul through `jnp.matmul` so SafeJnp can
            # neutralize 0-D operands inside chained expressions (`a @ b @ c`).
            eq_code = _rewrite_matmul_operator(eq_code)
        # Wir fügen die LIBS direkt in den globalen Scope des exec ein
        exec(eq_code, LIBS, namespace)

        # Filtere alle Funktionen heraus
        callables = {
            k: v for k, v in namespace.items()
            if callable(v) and not k.startswith("__")
        }

        if not callables:
            raise ValueError("Keine Funktion im eq_code gefunden.")

        func_name = list(callables.keys())[-1]
        func = callables[func_name]

        """
        def wrapper(*args):
            return func(*args)
        """
        #print("func", func)
        return func
    except Exception as e:
        print(f"Warn create_runnable: {type(e).__name__}: {e}")
        raise e


import inspect


def debug_callable(func):
    sig = inspect.signature(func)
    params = sig.parameters
    print(f"--- Debugging Callable: {func.__name__} ---")
    print(f"Anzahl Parameter (Signature): {len(params)}")
    print(f"Parameter Namen: {list(params.keys())}")

    # Prüfen auf Closures (versteckte Variablen)
    if hasattr(func, "__closure__") and func.__closure__:
        print(f"Anzahl versteckter Variablen (Closures): {len(func.__closure__)}")

    # Prüfen, ob es eine gebundene Methode ist (self-Problem)
    if hasattr(func, "__self__"):
        print("WARNUNG: Diese Funktion ist an ein Objekt gebunden (enthält 'self')!")
    return len(params)
