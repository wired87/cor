# COR — Compiler of Reality

Think of COR like a **science garden** for ideas:

```
        🌌 Discovery
           │
      🔬 Equation
           │
         🌳 Model
        /   |   \
      ⚛️  🧪  📈 Outcome
```

From the perspective of a top quantum scientist: this project helps turn scientific equations into a living simulation, so we can observe how complex systems might evolve over time.

## What this project does

- Takes equations and field definitions
- Builds a structured simulation graph
- Runs the simulation engine
- Produces data you can inspect and visualize

## Potential outcome

COR can help you explore “what could happen next” in complex scientific systems.  
In plain terms: it supports discovery, comparison of scenarios, and faster experiment ideas before expensive real-world testing.

## How to run

From the repository root:

```bash
python main.py
```

## Important note about equations and fields

The engine is designed to work with **all kinds of equations**, as long as:

1. the equations are provided in the `sm_manager/arsenal` directory, and  
2. required fields are integrated to `qfutils`. (todo make it kore user friendly)
