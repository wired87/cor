import jax
import jax.numpy as jnp

LIBS={
    "jax": jax,
    "vmap": jax.vmap,
    "jnp": jnp,
    "jit": jax.jit,
}

def create_runnable(eq_code):
    """
    Create runnable for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `eq_code`.
    2. Builds intermediate state such as `namespace`, `callables`, `func_name` before applying the main logic.
    3. Branches on validation or runtime state to choose the next workflow path.
    4. Delegates side effects or helper work through `exec()`, `ValueError()`, `list()`.
    5. Returns the assembled result to the caller.

    Inputs:
    - `eq_code`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    try:
        namespace = {}

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
        return func
    except Exception as e:
        print(f"Err core.qbrain.core.module_manager.create_runnable::create_runnable | handler_line=46 | {type(e).__name__}: {e}")
        print(f"[exception] core.qbrain.core.module_manager.create_runnable.create_runnable: {e}")
        print(f"Err create_runnable: {e}")
        raise e

