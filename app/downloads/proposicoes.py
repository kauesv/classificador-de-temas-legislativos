import wget

FORMATO = "json"
ANO = 2026

url = f'http://dadosabertos.camara.leg.br/arquivos/proposicoes/{FORMATO}/proposicoes-{ANO}.{FORMATO}'

output_directory = './docs/proposicoes'
filename = wget.download(url, out=output_directory)
