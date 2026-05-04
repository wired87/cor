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

import re
import jax.numpy as jnp
import jax

LIBS={
    "jax": jax,
    "vmap": jax.vmap,
    "jnp": jnp,
    "jit": jax.jit,
}


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
