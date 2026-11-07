"""Configuração da demonstração; o esquema científico será definido depois."""

from pathlib import Path
import tomllib


def raiz_projeto() -> Path:
    """Localiza o projeto a partir do diretório atual ou do pacote editável."""
    for inicio in (Path.cwd(), Path(__file__).resolve().parent):
        for diretorio in (inicio, *inicio.parents):
            if (diretorio / "pyproject.toml").is_file() and (diretorio / "configs").is_dir():
                return diretorio
    raise FileNotFoundError("Não foi possível localizar a raiz do projeto BILSTM.")


def carregar_configuracao(caminho: str | Path, parametros: dict | None = None) -> dict:
    """Carrega TOML, valida parâmetros e aplica substituições explícitas."""
    arquivo = Path(caminho)
    if not arquivo.is_absolute():
        arquivo = raiz_projeto() / arquivo
    with arquivo.open("rb") as entrada:
        config = tomllib.load(entrada)
    if set(config) != {"experimento"} or not isinstance(config["experimento"], dict):
        raise ValueError("A configuração deve conter somente a tabela [experimento].")
    valores = {"tipo": "demonstracao_sintetica", "seed": 42, "pontos": 12}
    chaves = set(valores)
    desconhecidas = set(config["experimento"]) - chaves
    if desconhecidas:
        raise ValueError(f"Parâmetros desconhecidos: {', '.join(sorted(desconhecidas))}")
    valores.update(config["experimento"])
    substituicoes = {} if parametros is None else parametros
    if not isinstance(substituicoes, dict):
        raise ValueError("Os parâmetros devem ser um dicionário.")
    desconhecidas = set(substituicoes) - {"seed", "pontos"}
    if desconhecidas:
        raise ValueError(f"Substituições desconhecidas: {', '.join(sorted(desconhecidas))}")
    valores.update(substituicoes)
    if valores["tipo"] != "demonstracao_sintetica":
        raise ValueError("Somente o tipo 'demonstracao_sintetica' está implementado.")
    if type(valores["seed"]) is not int or not 0 <= valores["seed"] < 2**32:
        raise ValueError("seed deve ser um inteiro entre 0 e 2^32 - 1.")
    if type(valores["pontos"]) is not int or not 2 <= valores["pontos"] <= 1000:
        raise ValueError("pontos deve ser um inteiro entre 2 e 1000.")
    return {"experimento": valores}
