"""Criação e consulta de execuções isoladas da demonstração sintética."""

import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import subprocess
from uuid import uuid4

import numpy as np
import tomli_w

from .configuracao import carregar_configuracao, raiz_projeto
from .graficos import grafico_demonstracao

_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")


def _caminho_execucao(identificador: str) -> Path:
    if not isinstance(identificador, str) or not _ID.fullmatch(identificador):
        raise ValueError("ID inválido: use letras, números, hífen ou sublinhado.")
    return raiz_projeto() / "output" / identificador


def _escrever_json(caminho: Path, dados: dict) -> None:
    caminho.write_text(json.dumps(dados, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _ambiente() -> dict:
    try:
        revisao = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=raiz_projeto(), check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        alteracoes = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=raiz_projeto(), check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        revisao = None
        alteracoes = None
    return {
        "python": platform.python_version(), "sistema": platform.system(),
        "arquitetura": platform.machine(), "revisao_git": revisao,
        "alteracoes_nao_commitadas": alteracoes,
        "pacotes": {nome: version(nome) for nome in ("numpy", "matplotlib", "bilstm-experimentos")},
    }


def executar(configuracao: str | Path, parametros: dict | None = None, *, identificador: str | None = None) -> str:
    """Executa exclusivamente a demonstração sintética e devolve seu ID."""
    config = carregar_configuracao(configuracao, parametros)
    if identificador is None:
        identificador = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    destino = _caminho_execucao(identificador)
    destino.mkdir(parents=True, exist_ok=False)
    manifesto = {
        "id": identificador, "estado": "em_andamento", "tipo": "demonstracao_sintetica",
        "origem": "dados sintéticos gerados para demonstração; não representam resultados científicos",
        "unidade": "observação ilustrativa", "inicio_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        _escrever_json(destino / "manifesto.json", manifesto)
        with (destino / "config_efetiva.toml").open("wb") as arquivo:
            tomli_w.dump(config, arquivo)
        _escrever_json(destino / "ambiente.json", _ambiente())
        valores = np.random.default_rng(config["experimento"]["seed"]).normal(
            size=config["experimento"]["pontos"]
        )
        with (destino / "dados_sinteticos.csv").open("w", encoding="utf-8", newline="") as arquivo:
            escritor = csv.writer(arquivo)
            escritor.writerow(["observacao", "valor_ilustrativo"])
            escritor.writerows(enumerate(valores, start=1))
        grafico_demonstracao(valores, destino / "figuras")
        manifesto.update(estado="concluida", fim_utc=datetime.now(timezone.utc).isoformat())
        (destino / "execucao.log").write_text("Demonstração sintética concluída.\n", encoding="utf-8")
        _escrever_json(destino / "manifesto.json", manifesto)
    except Exception as erro:
        manifesto.update(estado="falhou", fim_utc=datetime.now(timezone.utc).isoformat())
        _escrever_json(destino / "manifesto.json", manifesto)
        (destino / "execucao.log").write_text(f"Falha: {type(erro).__name__}.\n", encoding="utf-8")
        raise
    return identificador


def carregar(identificador: str) -> dict:
    """Lê uma execução existente sem executar novamente o experimento."""
    destino = _caminho_execucao(identificador)
    manifesto = json.loads((destino / "manifesto.json").read_text(encoding="utf-8"))
    if manifesto.get("estado") != "concluida":
        raise ValueError(f"Execução {identificador} não está concluída: {manifesto.get('estado')}.")
    config = carregar_configuracao(destino / "config_efetiva.toml")
    return {"manifesto": manifesto, "configuracao": config, "diretorio": destino}
