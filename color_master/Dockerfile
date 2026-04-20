FROM python:3.11-slim

WORKDIR /app

ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8080
ENV MCP_PATH=/mcp
ENV PYTHONUNBUFFERED=1

COPY r.txt .
RUN pip install --no-cache-dir -r r.txt

COPY main.py .
COPY mcp_server ./mcp_server

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/status')" || exit 1

CMD ["python", "mcp_server/routes.py"]
