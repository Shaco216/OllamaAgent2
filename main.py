#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lokaler Agent ohne API-Keys (läuft komplett lokal über Ollama)
================================================================

✅ Was du bekommst
- Einen Chat-Agent in Python mit:
  - **Lokalem LLM** via **Ollama** (z. B. `llama3`, `mistral`, `phi3`, `qwen2`)
  - **Kleiner Tool-Suite**: /calc, /save, /recall, /files, /clear, /persona, /help
  - **Persistenter Memory** (eine JSON-Datei)
  - **Konfigurierbarer Persona/System-Prompt**
  - **Kein API-Key, keine Cloud**

🛠️ Voraussetzungen
1) Python 3.9+
2) `pip install requests rich`
3) **Ollama** installieren: https://ollama.com
   - Nach Installation: ein Modell ziehen, z. B. `ollama pull llama3`
   - Starte den Ollama-Server: `ollama serve` (läuft meist automatisch im Hintergrund)

▶️ Starten
```
python local_agent.py --model llama3
```

💡 Beispieleingaben
- Normale Frage: `Erkläre mir kurz Quicksort.`
- Tool: `/calc (2+3)*4`
- Notiz speichern: `/save Merke: Morgen 10:00 Kundencall.`
- Memory ansehen: `/recall`
- Persona setzen: `/persona Du bist ein präziser, knapper Tech-Experte.`
- Hilfe: `/help`

Hinweis
- Dieses Skript nutzt **keine OpenAI- oder sonstige externen APIs** – nur den lokalen Ollama-Endpoint (http://localhost:11434).
- Es handelt sich um ein Referenzprojekt; passe es nach Bedarf an.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any

import requests
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table

console = Console()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
MEMORY_PATH = os.environ.get("AGENT_MEMORY", "agent_memory.json")
DEFAULT_SYSTEM_PROMPT = (
    "Du bist ein hilfreicher, genauer, faktenorientierter Assistent. "
    "Antworte knapp, strukturiert und in der Sprache des Nutzers. "
    "Wenn du dir unsicher bist, sag ehrlich, dass du es nicht weißt."
)

SUPPORTED_MODELS = [
    "llama3",
    "mistral",
    "phi3",
    "qwen2.5-coder:3b",
    "qwen2.5-coder:7b",
    "qwen2",
    "gemma2",
]


# ------------------------- Utility: Safe Eval for /calc -------------------------
import ast
import operator as op

ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}


def _eval_expr(node):
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.BinOp):
        if type(node.op) not in ALLOWED_OPS:
            raise ValueError("Operator nicht erlaubt")
        return ALLOWED_OPS[type(node.op)](_eval_expr(node.left), _eval_expr(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPS:
        return ALLOWED_OPS[type(node.op)](_eval_expr(node.operand))
    if isinstance(node, ast.Expression):
        return _eval_expr(node.body)
    raise ValueError("Ausdruck nicht erlaubt")


def safe_eval(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    return _eval_expr(tree)


# ---------------------------- Persistence (Memory) -----------------------------
@dataclass
class Memory:
    path: str = MEMORY_PATH
    data: Dict[str, Any] = field(default_factory=lambda: {"notes": [], "persona": DEFAULT_SYSTEM_PROMPT})

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                console.print("[yellow]Warnung: Konnte Memory nicht laden, starte frisch.")
        else:
            self.save()

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def add_note(self, note: str):
        self.data.setdefault("notes", []).append(note)
        self.save()

    def list_notes(self) -> List[str]:
        return self.data.get("notes", [])

    def clear(self):
        self.data = {"notes": [], "persona": DEFAULT_SYSTEM_PROMPT}
        self.save()

    def set_persona(self, text: str):
        self.data["persona"] = text.strip() or DEFAULT_SYSTEM_PROMPT
        self.save()

    def get_persona(self) -> str:
        return self.data.get("persona", DEFAULT_SYSTEM_PROMPT)


# ------------------------------ Ollama Chat Call -------------------------------

def ollama_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.2, top_p: float = 0.95, max_tokens: int | None = None) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens is not None:
        payload["options"] = {"num_predict": max_tokens}

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # Ollama's /api/chat returns a dict with 'message' at the end of stream mode
        if isinstance(data, dict) and "message" in data and isinstance(data["message"], dict):
            return data["message"].get("content", "")
        # Some versions may stream; for simplicity, if 'message' missing, try 'content'/best-effort
        if isinstance(data, dict) and "content" in data:
            return data["content"]
        return json.dumps(data, ensure_ascii=False)
    except requests.exceptions.ConnectionError:
        console.print("[red]Konnte nicht mit Ollama verbinden. Läuft der Server? (ollama serve)")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Ollama-Fehler: {e}")
        return "Fehler bei der Modellabfrage."


# --------------------------------- Agent Loop ---------------------------------
@dataclass
class Agent:
    model: str
    memory: Memory

    def build_system_prompt(self) -> str:
        persona = self.memory.get_persona()
        tool_doc = (
            "Du verfügst in dieser Laufzeitumgebung NICHT über Internetzugang. "
            "Du kannst einfache Mathematik intern lösen, aber für verlässliche Rechenschritte gibt es den /calc-Befehl. "
            "Wenn eine Anfrage externe Daten erfordern würde, erkläre offen die Grenzen."
        )
        return f"{persona}\n\n[Werkzeug-Hinweise]\n{tool_doc}"

    def reply(self, user_input: str) -> str:
        system = self.build_system_prompt()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_input},
        ]
        return ollama_chat(self.model, messages)


# -------------------------------- CLI / Tools ---------------------------------
HELP_TEXT = """
[bold]Verfügbare Befehle[/bold]
/help                 – diese Hilfe anzeigen
/calc <expr>          – sicheren Taschenrechner nutzen (z. B. /calc (2+3)*4)
/save <text>          – Notiz speichern
/recall               – alle Notizen anzeigen
/files                – Speicherdatei & Pfad anzeigen
/clear                – Memory leeren (Notizen & Persona)
/persona <text>       – Persona/System-Prompt setzen
/quit                 – Programm beenden

Ohne Slash-Befehl wird deine Eingabe an das Modell geschickt.
"""


def print_header(model: str, memory_path: str):
    console.print(Panel.fit(f"[bold]Lokaler Python-Agent[/bold]\nModell: [cyan]{model}[/cyan]\nMemory: [magenta]{memory_path}[/magenta]", title="Agent gestartet", width=80))
    console.print(Panel(Markdown("**Tipps:** Nutze `/persona` zum Feinjustieren des Stils. `/help` zeigt alle Befehle.")))


def cmd_calc(expr: str):
    try:
        val = safe_eval(expr)
        console.print(f"= {val}")
    except Exception as e:
        console.print(f"[red]Ungültiger Ausdruck:[/red] {e}")


def cmd_recall(mem: Memory):
    notes = mem.list_notes()
    if not notes:
        console.print("[dim]Keine Notizen gespeichert.[/dim]")
        return
    table = Table(title="Notizen")
    table.add_column("#", justify="right")
    table.add_column("Inhalt", overflow="fold")
    for i, n in enumerate(notes, 1):
        table.add_row(str(i), n)
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Lokaler Agent ohne API-Keys (Ollama)")
    parser.add_argument("--model", default="llama3", help=f"Ollama-Modell ({', '.join(SUPPORTED_MODELS)} oder beliebig, falls installiert)")
    parser.add_argument("--memory", default=MEMORY_PATH, help="Pfad zur Memory-JSON")
    args = parser.parse_args()

    mem = Memory(path=args.memory)
    mem.load()

    agent = Agent(model=args.model, memory=mem)
    print_header(args.model, args.memory)

    while True:
        try:
            user_in = Prompt.ask("[bold green]Du[/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold]Bye![/bold]")
            break

        if not user_in.strip():
            continue

        if user_in.startswith("/quit"):
            console.print("[bold]Bye![/bold]")
            break
        elif user_in.startswith("/help"):
            console.print(Markdown(HELP_TEXT))
            continue
        elif user_in.startswith("/calc"):
            expr = user_in[len("/calc"):].strip()
            if not expr:
                expr = Prompt.ask("Ausdruck")
            cmd_calc(expr)
            continue
        elif user_in.startswith("/save"):
            note = user_in[len("/save"):].strip()
            if not note:
                note = Prompt.ask("Notiztext")
            mem.add_note(note)
            console.print("[green]Gespeichert.[/green]")
            continue
        elif user_in.startswith("/recall"):
            cmd_recall(mem)
            continue
        elif user_in.startswith("/files"):
            console.print(f"Memory-Datei: [magenta]{mem.path}[/magenta] ({os.path.abspath(mem.path)})")
            continue
        elif user_in.startswith("/clear"):
            mem.clear()
            console.print("[yellow]Memory geleert (Notizen & Persona zurückgesetzt).[/yellow]")
            continue
        elif user_in.startswith("/persona"):
            persona = user_in[len("/persona"):].strip()
            if not persona:
                persona = Prompt.ask("Neue Persona/System-Prompt")
            mem.set_persona(persona)
            console.print("[green]Persona aktualisiert.[/green]")
            continue
        else:
            # normal chat
            reply = agent.reply(user_in)
            console.print(Panel(Markdown(reply), title="Agent"))


if __name__ == "__main__":
    main()
