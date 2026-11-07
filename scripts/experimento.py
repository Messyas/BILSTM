"""Interface de terminal para a demonstração e a consulta de execuções."""

import argparse
import json

from bilstm_pipeline.execucao import carregar, executar


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimentos do projeto BILSTM")
    comandos = parser.add_subparsers(dest="comando", required=True)
    iniciar = comandos.add_parser("executar", help="Executa a demonstração sintética")
    iniciar.add_argument("--config", required=True, help="TOML relativo à raiz do projeto ou absoluto")
    iniciar.add_argument("--id", help="ID opcional; deve ser novo")
    iniciar.add_argument("--seed", type=int, help="Substituição explícita da seed")
    iniciar.add_argument("--pontos", type=int, help="Quantidade de observações sintéticas")
    mostrar = comandos.add_parser("mostrar", help="Consulta uma execução concluída")
    mostrar.add_argument("--id", required=True)
    args = parser.parse_args()
    try:
        if args.comando == "executar":
            parametros = {chave: getattr(args, chave) for chave in ("seed", "pontos") if getattr(args, chave) is not None}
            identificador = executar(args.config, parametros, identificador=args.id)
            print(f"Demonstração sintética concluída. ID: {identificador}")
        else:
            execucao = carregar(args.id)
            print(json.dumps({"manifesto": execucao["manifesto"], "configuracao": execucao["configuracao"]}, ensure_ascii=False, indent=2))
    except (ValueError, OSError) as erro:
        parser.exit(1, f"Erro: {erro}\n")


if __name__ == "__main__":
    main()
