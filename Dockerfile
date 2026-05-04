FROM python:3.11-slim

WORKDIR /app

ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8080
ENV MCP_PATH=/mcp
ENV MCP_TRANSPORT=streamable-http
ENV PYTHONUNBUFFERED=1

# Core runtime deps (FastMCP, numpy, matplotlib) live here.
COPY color_master/r.txt /app/r.txt
RUN pip install --no-cache-dir -r /app/r.txt

# Repo code needed for the wrapped workflow.
COPY main.py /app/main.py
COPY mcp.py /app/mcp.py
COPY cor /app/core
COPY color_master /app/color_master

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')" || exit 1

CMD ["python", "mcp.py"]

