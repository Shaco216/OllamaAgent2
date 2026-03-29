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
    data: Dict[str, Any] = field(default_factory=lambda: {
        "notes": [],
        "persona": DEFAULT_SYSTEM_PROMPT,
        "master_profile": []})

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

    def add_to_persona(self, text: str):
        """
        Hängt Text an die bestehende Persona an.
        """
        text = text.strip()
        if not text:
            return  # nichts zu tun

        current = self.data.get("persona", DEFAULT_SYSTEM_PROMPT)
        self.data["persona"] = current + "\n" + text
        self.save()

    def get_persona(self) -> str:
        return self.data.get("persona", DEFAULT_SYSTEM_PROMPT)

    def add_master_info(self, text: str):
        self.data.setdefault("master_profile", []).append(text)
        self.save()

    def get_master_info(self) -> List[str]:
        return self.data.get("master_profile", [])

    def clear_all(self):
        self.data = {
            "notes": [],
            "persona": DEFAULT_SYSTEM_PROMPT,
            "master_profile": []
        }
        self.save()

    def clear_notes(self):
        self.data["notes"] = []
        self.save()

    def clear_persona(self):
        self.data["persona"] = DEFAULT_SYSTEM_PROMPT
        self.save()

    def clear_master(self):
        self.data["master_profile"] = []
        self.save()

    def show_all_infos(self):
        table = Table(title="Alle gespeicherten Informationen", show_lines=True)
        table.add_column("Kategorie", style="cyan", no_wrap=True)
        table.add_column("Inhalt", style="magenta", overflow="fold")

        # Persona
        table.add_row("Persona", self.get_persona())

        # Notizen
        notes = self.list_notes()
        table.add_row("Notizen", "\n".join(notes) if notes else "[dim]Keine Notizen[/dim]")

        # Master/User-Profil
        master = self.get_master_info()
        table.add_row("User-Profil", "\n".join(master) if master else "[dim]Keine Infos[/dim]")

        return table


# ------------------------------ Ollama Chat Call -------------------------------

def ollama_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.2, top_p: float = 0.95, max_tokens: int | None = None) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False, # fix für Ollama-Fehler: Extra data: line 2 column 1 (char 122)
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
        master_info = self.memory.get_master_info()

        master_text = ""
        if master_info:
            master_text = "Bekannte Informationen über den Nutzer:\n- " + "\n- ".join(master_info)

        tool_doc = (
            "Du verfügst in dieser Laufzeitumgebung NICHT über Internetzugang. "
            "Du kannst einfache Mathematik intern lösen, aber für verlässliche Rechenschritte gibt es den /calc-Befehl. "
            "Wenn eine Anfrage externe Daten erfordern würde, erkläre offen die Grenzen."
        )

        return f"{persona}\n\n{master_text}\n\n[Werkzeug-Hinweise]\n{tool_doc}"

    def reply(self, user_input: str) -> str:
        system = self.build_system_prompt()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_input},
        ]
        return ollama_chat(self.model, messages)


# -------------------------------- CLI / Tools ---------------------------------
HELP_TEXT = """\
[green]
[bold]Verfügbare Befehle[/bold]

/help            – diese Hilfe anzeigen  
/calc <expr>     – sicheren Taschenrechner nutzen (z. B. /calc (2+3)*4)  
/files           – Speicherdatei & Pfad anzeigen  
/clear           – Memory komplett leeren
/show            - Zeigt alle gespeicherten Infos  

/persona <text>  – Persona/System-Prompt setzen  
/showPersona     – aktuelle Persona anzeigen  
/clearPersona    – setzt die Persona zurück
/learnPersona    - Informationen über Persona hinzufügen  

/learn <text>    – Informationen über dich speichern  
/showMaster      – gespeicherte Infos über dich anzeigen  
/clearMaster     – löscht alle Infos über dich  

/save <text>     – Notiz speichern  
/showNotes       – alle Notizen anzeigen  
/clearNotes      – löscht alle Notizen  

/quit            – Programm beenden  

Ohne Slash-Befehl wird deine Eingabe an das Modell geschickt.
[/green]
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


def show_notes(mem: Memory):
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
            console.print(HELP_TEXT, markup=True)
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
        elif user_in.startswith("/showNotes"):
            show_notes(mem)
            continue
        elif user_in.startswith("/files"):
            console.print(f"Memory-Datei: [magenta]{mem.path}[/magenta] ({os.path.abspath(mem.path)})")
            continue
        elif user_in.startswith("/clear"):
                mem.clear_all()
                console.print("[yellow]Alles gelöscht (Notes, Persona, Master).[/yellow]")

        elif user_in.startswith("/clearNotes"):
                mem.clear_notes()
                console.print("[yellow]Notizen gelöscht.[/yellow]")

        elif user_in.startswith("/clearPersona"):
                mem.clear_persona()
                console.print("[yellow]Persona zurückgesetzt.[/yellow]")

        elif user_in.startswith("/clearMaster"):
                mem.clear_master()
                console.print("[yellow]User-Profil gelöscht.[/yellow]")
        elif user_in.startswith("/persona"):
            persona = user_in[len("/persona"):].strip()
            if not persona:
                persona = Prompt.ask("Neue Persona/System-Prompt")
            mem.set_persona(persona)
            console.print("[green]Persona aktualisiert.[/green]")
            continue
        elif user_in.startswith("/learnPersona"):
            persona_add = user_in[len("/learnPersona"):].strip()
            if not persona_add:
                persona_add = Prompt.ask("Erzähl mir mehr über mich")
            mem.add_to_persona(persona_add)
            console.print("[green]Persona aktualisiert.[/green]")
        elif user_in == "/learn":
            info = user_in[len("/learn"):].strip()
            if not info:
                info = Prompt.ask("Was soll ich über dich lernen?")
            mem.add_master_info(info)
            console.print("[green]Gespeichert (User-Profil).[/green]")
            continue
        elif user_in.startswith("/showMaster"):
            info = mem.get_master_info()
            if not info:
                console.print("[dim]Keine Infos über dich gespeichert.[/dim]")
            else:
                table = Table(title="User-Profil")
                table.add_column("#", justify="right")
                table.add_column("Info", overflow="fold")
                for i, n in enumerate(info, 1):
                    table.add_row(str(i), n)
                console.print(table)
            continue
        elif user_in.startswith("/showPersona"):
            persona = mem.get_persona()
            console.print(Panel(persona, title="Aktuelle Persona"))
            continue
        elif user_in == "/show":
            console.print(mem.show_all_infos())
        else:
            # normal chat
            reply = agent.reply(user_in)
            console.print(Panel(Markdown(reply), title="Agent"))


if __name__ == "__main__":
    main()
