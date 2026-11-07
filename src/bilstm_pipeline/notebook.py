"""Interface compacta para os notebooks, compartilhando a API da CLI."""

from .execucao import carregar, executar


def mostrar_parametros(identificador: str) -> None:
    from IPython.display import display

    execucao = carregar(identificador)
    print(f"Execução: {identificador}")
    display(execucao["configuracao"])


def mostrar_resultados(identificador: str) -> None:
    from IPython.display import Image, display
    import pandas as pd

    execucao = carregar(identificador)
    print(execucao["manifesto"]["origem"])
    display(pd.read_csv(execucao["diretorio"] / "dados_sinteticos.csv"))
    display(Image(filename=str(execucao["diretorio"] / "figuras" / "demonstracao.png")))
