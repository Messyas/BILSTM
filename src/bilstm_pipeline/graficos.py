"""Estilo e exportação centralizados das figuras."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ESTILO = {
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "axes.titlesize": 12, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": "--", "savefig.dpi": 300,
}


def grafico_demonstracao(valores, destino: Path) -> None:
    """Salva figura ilustrativa em PDF vetorial e PNG; não altera rcParams."""
    destino.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(ESTILO):
        figura, eixo = plt.subplots(figsize=(17 / 2.54, 9 / 2.54), layout="constrained")
        try:
            eixo.plot(range(1, len(valores) + 1), valores, marker="o", color="#0173b2")
            eixo.set(title="Demonstração com dados sintéticos", xlabel="Observação", ylabel="Valor ilustrativo (u.a.)")
            eixo.spines["top"].set_visible(False)
            eixo.spines["right"].set_visible(False)
            figura.savefig(destino / "demonstracao.pdf")
            figura.savefig(destino / "demonstracao.png", dpi=300)
        finally:
            plt.close(figura)
