import wget

FORMATO = "json"
ANO = 2026

url = f'http://dadosabertos.camara.leg.br/arquivos/proposicoesTemas/{FORMATO}/proposicoesTemas-{ANO}.{FORMATO}'

output_directory = './docs/proposicoes/temas'
filename = wget.download(url, out=output_directory)
