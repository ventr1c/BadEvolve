"""Minimal stdio MCP reader: source IDs only, no arbitrary paths, writes or execution."""
import hashlib
import json
from pathlib import Path
import sys


class EvidenceReader:
    def __init__(self, workspace, receipts):
        self.root = Path(workspace).resolve()
        self.receipts = Path(receipts)
        self.manifest = json.loads((self.root / "manifest.json").read_text())

    def _text(self, source):
        entry = self.manifest["sources"][source]
        path = (self.root / entry["file"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Source escapes the evidence workspace")
        return path.read_text()

    def _record(self, source, text, full):
        record = {"source": source, "text": text,
                  "source_sha256": hashlib.sha256(full.encode()).hexdigest()}
        with self.receipts.open("a") as out:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def call(self, name, args):
        if name == "list_evidence":
            names = [s for s in self.manifest["sources"] if s.startswith(args.get("prefix", ""))]
            offset = max(0, int(args.get("offset", 0)))
            limit = min(max(int(args.get("limit", 50)), 1), 100)
            selected = names[offset:offset + limit]
            return {**self.manifest, "sources": {s: self.manifest["sources"][s] for s in selected},
                    "source_count": len(names),
                    "next_offset": offset + len(selected) if offset + len(selected) < len(names) else None}
        if name == "read_evidence":
            source = args["source"]
            text = self._text(source)
            offset = int(args.get("offset", 0))
            limit = min(max(int(args.get("limit", 16000)), 1), 32000)
            if offset < 0 or offset > len(text):
                raise ValueError("Invalid offset")
            chunk = text[offset:offset + limit]
            return {**self._record(source, chunk, text), "offset": offset,
                    "total_chars": len(text),
                    "next_offset": offset + len(chunk) if offset + len(chunk) < len(text) else None}
        if name == "search_evidence":
            query = args["query"]
            if not isinstance(query, str) or not query:
                raise ValueError("A nonempty literal query is required")
            matches = []
            sources = [args["source"]] if args.get("source") else self.manifest["sources"]
            for source in sources:
                text = self._text(source)
                for i, line in enumerate(text.splitlines(), 1):
                    if query.casefold() in line.casefold():
                        at = max(0, line.casefold().find(query.casefold()) - 200)
                        matches.append({**self._record(source, line[at:at + 2000], text), "line": i})
                        if len(matches) == 40:
                            return {"matches": matches, "truncated": True}
            return {"matches": matches, "truncated": False}
        raise ValueError("Unknown evidence tool")


def tools():
    specs = [
        ("list_evidence", "List allowed source IDs, optionally filtered by prefix; page with next_offset.",
         {"prefix": {"type": "string"}, "offset": {"type": "integer", "minimum": 0},
          "limit": {"type": "integer", "minimum": 1, "maximum": 100}}, []),
        ("read_evidence", "Read a source by its exact manifest ID; page with next_offset until null.",
         {"source": {"type": "string"}, "offset": {"type": "integer", "minimum": 0},
          "limit": {"type": "integer", "minimum": 1, "maximum": 32000}}, ["source"]),
        ("search_evidence", "Case-insensitive literal search in a source or this bundle; at most 40 lines.",
         {"query": {"type": "string"}, "source": {"type": "string"}}, ["query"])]
    return [{"name": name, "description": description,
             "inputSchema": {"type": "object", "properties": properties, "required": required,
                             "additionalProperties": False},
             "annotations": {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}}
            for name, description, properties, required in specs]


def main():
    reader = EvidenceReader(sys.argv[1], sys.argv[2])
    for line in sys.stdin:
        message = json.loads(line)
        if "id" not in message:
            continue
        method, params = message["method"], message.get("params", {})
        if method == "initialize":
            result = {"protocolVersion": params.get("protocolVersion", "2024-11-05"),
                      "capabilities": {"tools": {}}, "serverInfo": {"name": "gate-evidence", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": tools()}
        elif method == "tools/call":
            try:
                value = reader.call(params["name"], params.get("arguments", {}))
                result = {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}]}
            except (KeyError, ValueError, OSError, TypeError) as exc:
                result = {"isError": True, "content": [{"type": "text", "text": str(exc)}]}
        elif method == "ping":
            result = {}
        else:
            print(json.dumps({"jsonrpc": "2.0", "id": message["id"],
                              "error": {"code": -32601, "message": "Method not found"}}), flush=True)
            continue
        print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}), flush=True)


if __name__ == "__main__":
    main()
