import wget

FORMATO = "json"
ANO = 2026

url = f'http://dadosabertos.camara.leg.br/arquivos/proposicoesAutores/{FORMATO}/proposicoesAutores-{ANO}.{FORMATO}'

output_directory = './docs/proposicoes/autores'
filename = wget.download(url, out=output_directory)
