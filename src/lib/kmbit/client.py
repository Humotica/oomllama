import json
import os
import time
import requests
from datetime import datetime

# The KmBiT Presence Service (part of JIS Router)
KMBIT_API = "http://localhost:8100/kmbit/presence"

class KmBiT:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.current_location = None

    def enter(self, file_path, intent="reading"):
        """Signal that the agent is entering a file/context."""
        self.current_location = file_path
        self._broadcast("enter", file_path, intent)
        print(f"📍 [{self.agent_name}] Entered {file_path} ({intent})")

    def act(self, file_path, action="editing"):
        """Signal an active modification."""
        self._broadcast("act", file_path, action)
        print(f"🔨 [{self.agent_name}] Action in {file_path}: {action}")

    def leave(self):
        """Signal leaving the current context."""
        if self.current_location:
            self._broadcast("leave", self.current_location, "done")
            print(f"👋 [{self.agent_name}] Left {self.current_location}")
            self.current_location = None

    def where_am_i(self):
        """Get spatial awareness context."""
        try:
            # In a real implementation, this fetches from the router
            # For now, we simulate the 'spatial view'
            return {
                "here": self.current_location,
                "nearby": self._get_nearby_files(self.current_location),
                "recent_visitors": [
                    {"agent": "codex", "ago": "5m", "intent": "audit"},
                    {"agent": "root_ai", "ago": "1h", "intent": "big_lift"}
                ]
            }
        except Exception as e:
            return {"error": str(e)}

    def _broadcast(self, event, location, detail):
        """Push the event to the AETHER (JIS Router)."""
        payload = {
            "agent": self.agent_name,
            "event": event,
            "location": location,
            "detail": detail,
            "timestamp": time.time()
        }
        # Fire and forget (UDP style via HTTP)
        try:
            requests.post(KMBIT_API, json=payload, timeout=0.5)
        except:
            pass

    def _get_nearby_files(self, path):
        if not path: return []
        directory = os.path.dirname(path)
        try:
            return [f for f in os.listdir(directory) if f != os.path.basename(path)][:5]
        except:
            return []

# Usage Example:
# bot = KmBiT("gemini")
# bot.enter("src/oomllama.rs", "optimizing")
# context = bot.where_am_i()
# print(context)
