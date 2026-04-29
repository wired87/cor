# COR — Field Simulation Engine

> **What is this?**
> COR is a system that simulates how **energy flows and changes over time** inside a network of interconnected nodes — similar to how waves travel through water, or how electrical signals propagate through a circuit. You feed it an image or a set of values, it turns that into an energy pattern, runs a physics-inspired simulation, and produces animated 3D visualizations of the result.

---

## In plain language

Imagine dropping a stone into a pond. Ripples spread out from the impact point, bounce off edges, and gradually fade. COR does something conceptually similar, but in a mathematical/computational space:

1. **You provide an input** — an image, a configuration, or raw data.
2. **COR reads the energy in that input** — bright pixels, high values, and strong signals all become "energy injections."
3. **A network of nodes is built** — think of it as a grid of tiny sensors that can sense, store, and pass on energy.
4. **The simulation runs step-by-step through time** — at each tick, every node updates based on what its neighbors sent it, creating a wave-like evolution.
5. **The results are visualized** as color-coded 3D animations so you can *see* how the energy moved and transformed.
6. **Everything is accessible via a simple API** so AI agents or other tools can trigger runs and receive results programmatically.

---

## System Overview

```mermaid
graph TB
    subgraph Input["📥 Input"]
        IMG["Image / Data\n(any PNG or JSON)"]
    end

    subgraph Parse["🔍 Parse & Map"]
        PARSER["in_parser\nConverts pixels → energy values\nper spatial position"]
    end

    subgraph Graph["🕸️ Build Graph"]
        FG["firegraph\nCreates the node network\n(modules, fields, edges)"]
        SM["sm_manager\nInitialises the graph state\nfor this simulation run"]
    end

    subgraph Inject["⚡ Inject Energy"]
        INJ["injector\nAttaches energy patterns\nto the right nodes at the right times"]
    end

    subgraph Simulate["🔬 Simulate"]
        GUARD["Guard\nOrchestrates the simulation:\nconfigures modules & equations"]
        JAX["JaxGuard / GNN\nRuns the actual maths step-by-step\nusing JAX on CPU/GPU"]
    end

    subgraph Visualize["🎨 Visualize"]
        CM["color_master\nRenders 3D animated GIFs\nshowing energy evolution over time"]
    end

    subgraph Serve["🌐 Serve"]
        MCP["mcp.py\nMCP + HTTP server\nAI agents call POST /run"]
    end

    IMG --> PARSER
    PARSER --> INJ
    FG --> SM
    SM --> INJ
    INJ --> GUARD
    FG --> GUARD
    GUARD --> JAX
    JAX --> CM
    CM --> MCP
    MCP -.->|"trigger"| PARSER
```

---

## End-to-End Workflow

The diagram below shows every major stage from receiving a request to returning a result.

```mermaid
flowchart LR
    A(["Client / AI Agent\nPOST /run"])

    subgraph step1["1 · Parse Input"]
        B["in_parser\nimage → energy map\n(pixel brightness = injection strength)"]
    end

    subgraph step2["2 · Build Simulation Graph"]
        C["firegraph\ncreate nodes & edges"]
        D["sm_manager\ninitialise graph state"]
    end

    subgraph step3["3 · Attach Injections"]
        E["injector\nbind energy patterns\nto field nodes"]
    end

    subgraph step4["4 · Run Simulation"]
        F["Guard\nset up equations\n& module chain"]
        G["JaxGuard + GNN\nstep-by-step field\ndynamics on JAX"]
    end

    subgraph step5["5 · Visualize"]
        H["color_master\n3D PNGs + animated GIFs\nper field key"]
    end

    subgraph step6["6 · Return"]
        I["JSON response\nwith base64 images\n& simulation metadata"]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> A
```

---

## Key Components

| Component | What it does (plain English) |
|-----------|------------------------------|
| `in_parser.py` | Reads an image and converts each pixel's brightness into an "energy injection" value at that spatial position |
| `firegraph/` | Builds and manages the graph — the network of nodes, fields, and connections that the simulation runs on |
| `sm_manager/` | Initialises the graph into a ready state before the simulation starts |
| `injector.py` | Takes the energy map produced by the parser and attaches it to the correct nodes in the graph at the right time steps |
| `guard.py` | The main simulation conductor — reads equations, sets up the module chain, and drives the run |
| `jax_test/` | The mathematical engine: a JAX-powered neural-graph network that computes field evolution step by step |
| `color_master/` | Turns raw simulation numbers into beautiful 3D animated color visualizations |
| `mcp.py` | A lightweight HTTP + MCP server so AI agents (Claude, Gemini, OpenAI, …) can trigger runs via `POST /run` |

---

## The Simulation Loop (one step at a time)

```mermaid
sequenceDiagram
    participant T as Time Controller
    participant N as Node (field)
    participant G as GNN Engine
    participant DB as State Store

    loop For each time step t
        T->>G: begin_step(t)
        G->>N: gather neighbor features
        N->>G: current field state
        G->>G: apply equations\n(inject → filter → compute → shift)
        G->>DB: save result for step t
        DB->>T: step complete, advance t
    end
    DB-->>G: serialize final state
```

---

## Quick Start

### Run via Docker (recommended)

```bash
docker build -t cor .
docker run -p 8080:8080 cor
```

### Call the simulation

```bash
curl -X POST http://localhost:8080/run \
  -H "Content-Type: application/json" \
  -d '{
    "sim_spec": {
      "amount_nodes": 4,
      "sim_time": 10,
      "dims": 3
    }
  }'
```

### Check server health

```bash
curl http://localhost:8080/health
curl http://localhost:8080/status
```

---

## For AI Agents

COR exposes a single MCP tool called **`run`**:

```
Tool:  run
Route: POST /run
```

**Input schema:**

```json
{
  "sim_spec": {
    "amount_nodes": 4,
    "sim_time": 20,
    "dims": 3,
    "user_id": 1
  },
  "injection_file": { ... }
}
```

**Output:** JSON with simulation metadata and base64-encoded visualization images (PNG statics + GIF animations per field key).

Agents supported: **Claude**, **Gemini**, **OpenAI** — see [`AGENTS.md`](AGENTS.md) for details.

---

## Sub-projects

| Folder | Short description |
|--------|------------------|
| [`jax_test/`](jax_test/README.md) | JAX/Flax GNN simulation engine — the mathematical core |
| [`color_master/`](color_master/README.md) | 3D visualization module with its own MCP server |

---

## Philosophy

> *Every state carries the current moment into the next — only the change is computed.*

The grid is not the goal; it is a building block for discovering **persistent patterns over time**. The true aim is to identify those patterns and understand how energy injections shape the next time step.
