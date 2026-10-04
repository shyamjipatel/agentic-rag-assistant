"""Deterministic Ollama HTTP fixture, available only in the container-check override."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        message = {"role": "assistant", "content": ""}
        if "tools" in body:
            message["tool_calls"] = [{"function": {"name": "calculator", "arguments": {
                "operation": "multiply", "left": "24", "right": "3", "source_ids": [1],
            }}}]
        elif "question" in body.get("format", {}).get("properties", {}):
            message["content"] = json.dumps({"question": "Annual leave days over three years?"})
        else:
            tools = [item for item in body["messages"] if item["role"] == "tool"]
            text = "Employees receive 24 annual leave days."
            if tools:
                result = json.loads(tools[-1]["content"])
                text = f"Employees receive {result['value']} leave days over three years."
            message["content"] = json.dumps({"supported": True, "statements": [
                {"text": text, "source_ids": [1]},
            ]})
        payload = json.dumps({"done": True, "done_reason": "stop", "message": message}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 11434), Handler).serve_forever()
