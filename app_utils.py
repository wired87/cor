import os
import sys
from pathlib import Path
import dotenv

from qfu.all_subs import ALL_SUBS

dotenv.load_dotenv()

MAX_GRID_SIZE = 3**3
SCHEMA_GRID = [
    (i, i, i)
    for i in range(MAX_GRID_SIZE)
]

_CORE_DIR = Path(__file__).resolve().parent
# Standard-model module sources live under sm_manager/arsenal (not cor/arsenal).
ARSENAL_PATH = str((_CORE_DIR / "sm_manager" / "arsenal").resolve())

## GCP
GCP_ID = os.environ.get("GCP_PROJECT_ID")
FBDB_INSTANCE = os.environ.get("FIREBASE_RTDB")
GEM_API_KEY=os.environ.get("GEMINI_API_KEY")
LOGGING_DIR = os.environ.get("LOGGING_DIR")


BASE_MODULES={}
def get_runnables():
    """
    Retrieve runnables for the surrounding workflow.

    Workflow:
    1. Starts from the current object state and local workflow context.
    2. Builds intermediate state such as `mock_jax` before applying the main logic.
    3. Delegates side effects or helper work through `MagicMock()`.
    4. Returns the assembled result to the caller.

    Inputs:
    - None.

    Returns:
    - Returns the computed result for the caller.
    """
    try:
        import jax.numpy as jnp
        import jax
        return {
            "jnp": jnp,
            "jit": jax.jit,
            "jax": jax,
            "vmap": jax.vmap,
        }
    except ImportError:
        print("Err cor.qbrain.cor.app_utils::get_runnables | handler_line=52 | ImportError handler triggered")
        print("[exception] cor.qbrain.cor.app_utils.get_runnables: caught ImportError")
        import numpy as np
        from unittest.mock import MagicMock
        mock_jax = MagicMock()
        mock_jax.numpy = np
        sys.modules["jax"] = mock_jax
        sys.modules["jax.numpy"] = np
        return {
            "jnp": np,
            "jit": lambda x, *args, **kwargs: x,
            "jax": mock_jax,
            "vmap": lambda x, *args, **kwargs: x,
        }


RUNNABLE_MODULES=get_runnables()

def get_demo_env():
    """
    Retrieve demo env for the surrounding workflow.

    Workflow:
    1. Starts from the current object state and local workflow context.
    2. Returns the assembled result to the caller.

    Inputs:
    - None.

    Returns:
    - Returns the computed result for the caller.
    """
    return {'cluster_dim': [12, 12, 12],
     'cpu': 6,
     'cpu_limit': 6,
     'device': 'cpu',
     'env': [
         {'name': 'SESSION_ID',
          'value': 'env_rajtigesomnlhfyqzbvx_atqbmkasfekqjfgkzrdr'},
         {'name': 'DOMAIN', 'value': 'bestbrain.tech'},
         {'name': 'GCP_ID', 'value': 'aixr-401704'},
         {'name': 'DATASET_ID', 'value': 'QCOMPS'},
         {'name': 'SIM_LEN_S', 'value': '300'},
         {'name': 'LOGGING_DIR', 'value': 'tmp/ray'},
         {'name': 'ENV_ID',
          'value': 'env_rajtigesomnlhfyqzbvx_atqbmkasfekqjfgkzrdr'},
         {'name': 'USER_ID', 'value': 'rajtigesomnlhfyqzbvx'},
         {'name': 'FIREBASE_RTDB',
          'value': 'https://bestbrain-39ce7-default-rtdb.firebaseio.com/'},
         {'name': 'FB_DB_ROOT',
          'value': 'users/rajtigesomnlhfyqzbvx/env/env_rajtigesomnlhfyqzbvx_atqbmkasfekqjfgkzrdr'},
         {'name': 'DELETE_POD_ENDPOINT', 'value': 'gke/delete-pod/'},
         {'name': 'GKE_SIM_CLUSTER_NAME', 'value': 'sims'}
     ],
     'gpus': 0,
     'mem': '12Gi',
     'mem_limit': '12Gi',
     'sim_time': 300
}
USER_ID = os.environ.get("USER_ID")
ENV_ID = os.environ.get("ENV_ID")
# sg
SG_IID = os.environ.get("ENV_ID")
SG_DBID = os.environ.get("SG_DB_ID")
GNAME = os.environ.get("ENV_ID")
DB_NAME=USER_ID
TABLE_NAME=ENV_ID

# ENV VARS
DOMAIN = os.environ.get("DOMAIN")
MODULE_PATH=os.environ.get("MODULE_PATH", r"sm_manager/sm")
SESSION_ID = os.environ.get("SESSION_ID")


# VARS
TESTING = True
FB_DB_ROOT = f"users/{USER_ID}/env/{ENV_ID}"
DEMO_ENV=get_demo_env()
HEAD_SERVER_NAME = "HEAD"
ENDPOINTS = [*ALL_SUBS, "EDGES"]
USED_ENDPOINTS = []
LOCAL_DATASTORE = False

JAX_DEVICE="cpu" if TESTING is True else "gpu"

GPU=os.getenv("GPU")

# SET ENVS
os.environ["APP_ROOT"] = os.path.dirname(os.path.abspath(sys.argv[0]))

NUM_GPU_TOTAL = 0 if TESTING is True else 1
NUM_GPU_NODE = 0 if TESTING is True else .33

NUM_CPU_TOTAL = 4

###########################
###### CLASSES

# LOGIC (related) VARS
if os.name == "nt":
    trusted = ["*"]
else:
    trusted=[
        f"{DOMAIN}.com", f"*.{DOMAIN}.com", "localhost", "127.0.0.1"]

ALL_DB_WORKERS=[
    "FBRTDB",
    "SPANNER_WORKER",
    "BQ_WORKER",
]


SIMULATE_ON_QC=os.getenv("SIMULATE_ON_QC")
if str(SIMULATE_ON_QC) == "0":
    SIMULATE_ON_QC = True
else:
    SIMULATE_ON_QC = False


GLOBAC_STORE = {
    key: None
    for key in [
        "UTILS_WORKER",
        "DB_WORKER",
        "HEAD",
        "GLOB_LOGGER",
        "GLOB_STATE_HANDLER",
        "BQ_WORKER",
        "SPANNER_WORKER",
        "WEB_DATA_PROVIDER",
    ]
}

def extend_globs(key, value):
    """
    Extend globs for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `key`, `value`.
    2. Delegates side effects or helper work through `print()`.
    3. Finishes by updating state, triggering side effects, or completing the workflow without a direct return value.

    Inputs:
    - `key`: Caller-supplied value used during processing.
    - `value`: Caller-supplied value used during processing.

    Returns:
    - Returns `None`; the main effects happen through state updates, I/O, or delegated calls.
    """
    GLOBAC_STORE[key] = value
    print(f"EXTEND GLOB STORE WITH {key}={value}")

def get_endpoint():
    """
    Retrieve endpoint for the surrounding workflow.

    Workflow:
    1. Starts from the current object state and local workflow context.
    2. Branches on validation or runtime state to choose the next workflow path.
    3. Delegates side effects or helper work through `USED_ENDPOINTS.append()`.
    4. Returns the assembled result to the caller.

    Inputs:
    - None.

    Returns:
    - Returns the computed result for the caller.
    """
    for endp in ENDPOINTS:
        if endp not in USED_ENDPOINTS:
            USED_ENDPOINTS.append(endp)
            return endp





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


