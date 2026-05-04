"""Test create route and MCP server."""
import importlib.util
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

spec = importlib.util.spec_from_file_location("routes", root / "mcp_server" / "routes.py")
routes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routes)
create = routes.create

# Minimal SOA dict
data = {"a": [[1, 2, 3], [4, 5, 6]], "b": [1.0, 2.0]}
r = create(data=data, amount_nodes=10, dims=10, output_dir="output_dir_test", use_demo_if_empty=False)
print(r)
assert r.get("ok") is True
assert "output_dir_test" in r.get("out", "")
print("test_mcp.py passed")
