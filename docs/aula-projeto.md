# Aula: como funciona o Classificador de Temas Legislativos

Material de leitura sobre o projeto e Machine Learning (ML)

## 1. O que este projeto faz

Este projeto recebe o texto de uma ementa de proposição legislativa e estima qual tema legislativo oficial está mais associado a ela. A aplicação oferece essa previsão por uma API HTTP feita com FastAPI. Um exemplo de entrada é `{"texto":"Dispõe sobre o acesso à educação básica nas escolas públicas."}`; uma resposta possível é `{"tema":"Educação","confianca":0.9909}`.

O propósito é ajudar na triagem e na busca. A previsão não é uma classificação oficial, não interpreta o mérito jurídico e não garante que todos os temas secundários de uma proposição sejam apresentados. O número chamado `confianca` é uma probabilidade estimada pelo modelo, não uma garantia de acerto.

Em termos de produto, é uma aplicação de ML supervisionado para classificação de texto. O treino usa exemplos anteriores em que já conhecemos o texto e os temas atribuídos pela Câmara. O modelo aprende regularidades entre palavras e rótulos e depois aplica essas regularidades a um texto novo.

## 2. O mapa do projeto

| Caminho | Responsabilidade |
|---|---|
| `app/baixar_dados.py` | Baixa arquivos anuais de proposições e temas da API de Dados Abertos da Câmara. |
| `app/dataset.py` | Lê os arquivos, junta proposições e temas pelo identificador, limpa ementas e escreve o dataset em CSV. |
| `app/train.py` | Seleciona temas, divide os dados, transforma texto em números, treina e compara classificadores, mede resultados e salva o modelo e relatórios. |
| `app/inference.py` | Usa o modelo carregado para pontuar os temas e montar a resposta com o tema de maior pontuação. |
| `app/main.py` | Define a API FastAPI, valida requisições, carrega o modelo e expõe `/health` e `/predict`. |
| `tests/` | Verifica o download, a preparação de dados, partes do treino e os endpoints. |
| `data/` | Local esperado para arquivos baixados e dados processados; os dados reais são excluídos do Git. |
| `models/theme_classifier.joblib` | Artefato serializado do modelo escolhido e dos metadados dos rótulos. |
| `outputs/` | Métricas, comparação de modelos, revisão de erros e banco local de tracking do MLflow. |
| `Dockerfile`, `docker-compose.yml` | Montam e executam a API dentro de um contêiner. |
| `requirements.txt` | Declara as bibliotecas Python usadas pela aplicação e pelo treino. |
| `README.md` | Explica a execução, os dados, a arquitetura e os resultados registrados. |

### Fluxo completo

```mermaid
flowchart LR
    A[Dados Abertos da Câmara] --> B[baixar_dados.py]
    B --> C[JSON anual de proposições e temas]
    C --> D[dataset.py: junção, validação e limpeza]
    D --> E[CSV com texto e rótulos]
    E --> F[train.py: divisão e treino]
    F --> G[TF-IDF + classificador]
    G --> H[Modelo .joblib e relatórios]
    H --> I[main.py carrega o modelo]
    I --> J[POST /predict]
    J --> K[inference.py escolhe maior pontuação]
```

Há dois ciclos de vida diferentes: o ciclo offline de preparação e treino, que pode ser demorado, e o ciclo online de inferência, que atende às requisições. A API não treina o modelo ao receber uma requisição: ela carrega um artefato que já foi treinado.

## 3. O que acontece em cada etapa

### 3.1 Download — `app/baixar_dados.py`

O script baixa dois conjuntos por ano: proposições e suas associações a temas. Os arquivos JSON são guardados em `data/proposicoes/` e `data/proposicoes/temas/`. Por padrão, um arquivo que já existe é mantido; `--sobrescrever` pede para baixá-lo de novo. Antes de gravar, o programa verifica se a resposta tem o formato JSON esperado. Ele grava primeiro em um arquivo temporário e só então o move para o nome definitivo, reduzindo a chance de deixar um arquivo parcial.

O downloader usa `urllib` da biblioteca padrão e timeout de 120 segundos. Esse é um script de ingestão: busca dados numa fonte externa e os traz para o projeto.

### 3.2 Preparação — `app/dataset.py`

Os arquivos de proposições contêm, entre outros campos, o ID, o ano e a ementa. Os arquivos de temas associam temas a uma proposição por meio da URI. O código extrai o último segmento da URI como ID e faz a junção por esse identificador. Junção por chave é uma operação comum de banco de dados: junta as informações que se referem à mesma entidade.

Depois, o código:

1. valida o formato de cada arquivo;
2. detecta IDs repetidos entre anos;
3. detecta um código oficial de tema que apareça com nomes conflitantes;
4. normaliza espaços da ementa e remove as ementas vazias;
5. descarta proposições sem tema associado;
6. remove associações repetidas entre uma mesma proposição e um mesmo tema;
7. grava um CSV e um resumo JSON com contagens.

O dataset fica com uma linha por proposição. As colunas centrais são `id`, `ano`, `texto`, `codigos_tema` e `temas`. Os dois últimos campos são listas codificadas como JSON dentro das células CSV. A limpeza dos dados é parte importante do sistema: erros ou inconsistências aqui viram exemplos ruins para o treino.

### 3.3 Treino e avaliação — `app/train.py`

O treinamento lê o CSV, remove linhas incompletas, interpreta as listas de rótulos, escolhe até dez temas frequentes com pelo menos cinquenta exemplos e transforma os rótulos em uma matriz de zeros e uns. O limite de cinco a dez temas e os mínimos são parâmetros do script, não propriedades universais de ML.

Em seguida, separa os dados em treino, validação e teste. Os tamanhos pretendidos são 65%, 15% e 20%. A divisão é agrupada pela ementa normalizada: textos iguais não podem cair em conjuntos diferentes. Isso evita um tipo de vazamento em que o mesmo texto aparece no treino e no teste, fazendo o resultado parecer melhor do que seria em exemplos realmente novos.

O pipeline compara duas alternativas:

- TF-IDF com regressão logística;
- TF-IDF com SVM linear (`LinearSVC`), cujos resultados são calibrados para produzir valores interpretáveis como probabilidades.

Ambas usam `OneVsRestClassifier`, uma estratégia para resolver um problema multilabel com classificadores binários. O código treina cada alternativa no conjunto de treino, escolhe a melhor pela macro-F1 de validação, depois volta a treinar com treino + validação e calcula as métricas finais no teste reservado. O conjunto de teste não deve orientar escolhas repetidas de parâmetros; se isso acontecer, ele deixa de ser uma avaliação imparcial.

O modelo selecionado é gravado em `models/theme_classifier.joblib`. O processo também grava métricas JSON, comparação CSV, amostra de erros e registros do MLflow em `outputs/`.

### 3.4 Inferência — `app/inference.py`

`predict_text` chama `predict_proba` do pipeline passando uma lista com o texto. O vetor retornado contém uma pontuação por tema. A função encontra o índice da maior pontuação, usa esse índice para localizar o código do tema e, em seguida, o nome legível. A resposta arredonda a pontuação para quatro casas decimais.

Ponto importante: embora o treino e as métricas também tratem o problema como multilabel, a função de inferência pública retorna apenas o tema de maior pontuação. Ela não retorna todos os temas que superam o limiar de 0,5.

### 3.5 API — `app/main.py`

`PredictionRequest` define o formato esperado no corpo da requisição: um campo `texto`, com entre 1 e 20.000 caracteres; espaços nas extremidades são removidos e texto que fica vazio é rejeitado. FastAPI e Pydantic validam a entrada. Uma requisição inválida normalmente recebe HTTP 422.

`GET /health` indica se o servidor está online e se conseguiu carregar o modelo. `POST /predict` delega a classificação a `predict_text`. Se o arquivo do modelo não existir, `/predict` responde com HTTP 503, indicando que o serviço ainda não está pronto para prever. O bundle carregado é guardado em `app.state.model_bundle` para não reler o arquivo a cada chamada.

O carregamento usa `joblib`, que desserializa objetos Python. Por isso, o arquivo do modelo deve vir de uma fonte confiável e compatível com o ambiente que executa a aplicação.

### 3.6 Execução — Docker

O `Dockerfile` instala as dependências, copia o projeto, cria um usuário sem privilégios administrativos e inicia Uvicorn com três workers. O Compose publica a porta 8001 do computador na porta 8000 do contêiner e define o caminho do artefato via `MODEL_PATH`.

Cada worker é um processo separado; portanto, cada processo pode carregar sua própria cópia do modelo na memória. Para um modelo pequeno isso simplifica o serviço; em ambientes com memória restrita, o consumo total deve ser considerado.

## 4. Conceitos de ML que aparecem aqui

### Aprendizado supervisionado

Aprender usando exemplos com respostas conhecidas. Neste caso, uma amostra é uma ementa (entrada) junto dos temas oficiais (resposta/target). O modelo ajusta seus parâmetros para associar padrões de texto a esses rótulos.

### Classificação

Prever uma categoria discreta, como “Saúde” ou “Educação”. Não é regressão no sentido habitual de prever uma quantidade contínua. “Regressão logística” tem a palavra regressão por sua origem estatística, mas aqui é usada como classificador.

### Rótulo, classe e target

São nomes para a resposta que se deseja prever. Neste projeto, os rótulos são os temas. “Classe” também é usada para uma categoria possível. `target` é o nome frequente para a saída correta usada no treino.

### Multiclasse e multilabel

Em multiclasse, cada exemplo pertence a exatamente uma categoria. Em multilabel, um exemplo pode ter vários rótulos simultâneos. As proposições oficiais podem ter vários temas, então os dados de treino são multilabel. O código cria um vetor por proposição, por exemplo `[1, 0, 1, 0]`, onde 1 indica associação ao tema daquela posição.

Mesmo assim, o endpoint atual devolve só um tema: o de maior pontuação. Isso é uma decisão de produto da API, não uma limitação da representação de treino. Em avaliação, o código também usa um limiar de 0,5 para decidir independentemente se cada rótulo é positivo; no endpoint, usa `argmax` (o índice do maior valor).

### Feature e representação numérica

Um algoritmo clássico não recebe diretamente uma frase em português: ele precisa de números. Uma *feature* é uma característica de entrada usada para aprender. TF-IDF transforma palavras e grupos de palavras em valores numéricos, criando uma representação esparsa: cada texto tende a usar apenas uma pequena parte do vocabulário conhecido.

### TF-IDF

TF-IDF é a combinação de frequência do termo e raridade do termo no conjunto. Uma palavra ganha peso quando aparece no texto, mas palavras presentes em quase todos os textos tendem a ser menos informativas. O `TfidfVectorizer` do projeto remove acentos para a representação e usa `ngram_range=(1, 2)`: considera palavras isoladas e pares consecutivos, como `saúde` e `saúde pública`. `min_df=2` ignora termos presentes em apenas um documento; `max_features=100_000` limita o vocabulário. São escolhas de representação, não compreensão semântica: o modelo não entende o significado jurídico da frase.

### Pipeline

Um pipeline encadeia as etapas para que o mesmo pré-processamento seja usado no treino e na previsão. Aqui, primeiro TF-IDF transforma texto; depois um classificador estima os rótulos. Salvar o pipeline completo ajuda a evitar que a API use uma transformação diferente daquela usada no treino.

### One-vs-Rest (um contra o resto)

Estratégia para multilabel: treinar um classificador binário para cada tema. Para “Saúde”, por exemplo, os exemplos com esse rótulo são positivos e os demais negativos. No final há uma pontuação independente por tema. Como os classificadores são independentes, pontuações altas em temas parecidos podem resultar em temas extras.

### Regressão logística

Classificador linear que combina os valores das features com pesos aprendidos e transforma a soma em uma probabilidade estimada. `class_weight="balanced"` aumenta o peso relativo de classes menos frequentes durante o ajuste, tentando reduzir o domínio dos rótulos comuns. Isso não resolve automaticamente todos os desequilíbrios.

### SVM linear e calibração

Uma SVM linear procura uma fronteira que separe exemplos positivos e negativos com uma margem ampla. `LinearSVC` produz scores de decisão, que não são probabilidades diretamente. `CalibratedClassifierCV(method="sigmoid")` ajusta uma transformação para converter scores em probabilidades aproximadas; a validação cruzada (`cv=3`) ajuda a fazer essa calibração. Uma probabilidade calibrada continua sendo estimativa, e sua qualidade depende dos dados.

### Treino, validação e teste

- **Treino:** ajusta os pesos/parâmetros do modelo.
- **Validação:** compara opções e orienta a escolha do modelo.
- **Teste:** estima o desempenho final em dados que não participaram das escolhas.

Uma semente aleatória (`random_state=42`) torna a divisão reproduzível para o mesmo dataset e ambiente. Reprodutibilidade não significa que todos os ambientes e versões de bibliotecas produzirão bits idênticos.

### Vazamento de dados

Ocorre quando informação do conjunto de avaliação influencia o treino ou a escolha do modelo, ou quando exemplos praticamente repetidos aparecem em treino e teste. Isso infla as métricas. O projeto agrupa ementas normalizadas para impedir que textos idênticos atravessem as divisões.

### Métricas: precisão, recall, F1

Para cada tema, pense em “positivo” como “este tema deve ser associado a esta proposição”.

- **Precisão:** entre as previsões positivas, quantas estavam corretas? Baixa precisão significa temas extras (falsos positivos).
- **Recall (revocação):** entre os casos positivos reais, quantos foram encontrados? Baixo recall significa temas omitidos (falsos negativos).
- **F1:** média harmônica de precisão e recall; baixa se qualquer uma das duas for baixa.
- **Macro-F1:** calcula F1 por tema e tira média simples, dando peso igual a temas frequentes e raros. É a métrica usada para escolher o modelo na validação.
- **Balanced accuracy:** para cada classificação binária de tema, faz a média entre recall dos positivos e recall dos negativos; depois o código tira a média dos temas. Ajuda a observar o desempenho com rótulos desbalanceados.
- **Top-1:** verifica se o tema com maior pontuação está entre os rótulos oficiais da proposição. É mais próximo do comportamento atual do endpoint, que devolve apenas um tema.
- **Suporte:** quantidade de exemplos positivos reais daquele tema na avaliação.

Na matriz de confusão, `TP` é verdadeiro positivo, `FP` falso positivo, `FN` falso negativo e `TN` verdadeiro negativo. Esses valores ajudam a entender por que uma métrica subiu ou caiu.

### Limiar e `argmax`

O limiar 0,5 significa: marcar um tema como positivo se sua estimativa for pelo menos 0,5. Como a decisão é independente por classe, podem sair zero, um ou vários temas. `argmax` significa escolher o índice da maior pontuação; ele sempre escolhe exatamente um tema (desde que existam classes). Assim, `top-1` e a seleção por limiar respondem a perguntas diferentes.

### Generalização, distribuição e viés

Generalizar é manter desempenho em exemplos novos. As métricas registradas no README referem-se a dados de 2026 e uma divisão aleatória por texto; não demonstram por si só desempenho em outros anos. Regras, linguagem e frequência de temas podem mudar. Uma avaliação temporal — treinar nos anos anteriores e testar em um ano posterior — seria necessária para avaliar melhor esse cenário.

O modelo também aprende os padrões dos rótulos existentes. Se a rotulagem histórica for inconsistente, ou se certos temas tiverem menos exemplos, as previsões podem reproduzir essas lacunas. A previsão é uma recomendação estatística que merece revisão humana em decisões relevantes.

## 5. Como interpretar os resultados atuais

O README informa 9.742 amostras selecionadas, separadas em 6.817 para treino, 979 para validação e 1.946 para teste. A regressão logística ficou com macro-F1 de validação de aproximadamente 0,726 e macro-F1 de teste de aproximadamente 0,728. O Top-1 de teste informado é aproximadamente 0,841. Esses números descrevem aquele recorte e aquela divisão; não são uma promessa de acerto para toda ementa futura.

Uma métrica agregada esconde diferenças entre temas: por exemplo, no relatório há F1 mais alto em Educação e Saúde e mais baixo em Indústria, Comércio e Serviços. Por isso, inspecione também os resultados por classe e os erros concretos em `docs/analise-erros.md` e `outputs/error_review.csv`. Um erro pode ser compreensível quando a ementa mistura assuntos próximos, mas isso não o torna correto.

## 6. Estado local e reprodução

No estado observado durante a preparação desta aula, o artefato `models/theme_classifier.joblib` e relatórios em `outputs/` existem no workspace. Já a pasta `data/processed/` contém apenas `.gitkeep`; não foi encontrado o `dataset.csv`, nem os JSON brutos. Esses artefatos são ignorados pelo Git. Portanto, a API pode estar pronta para inferência se o artefato local for legível, mas o treino completo não pode ser reproduzido sem baixar os dados e gerar o CSV primeiro.

Os comandos previstos pelo README, executados da raiz do repositório com o ambiente Python configurado, são:

```bash
pip install -r requirements.txt
python -m app.baixar_dados --inicio 2016 --fim 2026
python -m app.dataset --top-classes 10 --min-samples 50
python -m app.train
python -m uvicorn app.main:app --reload --port 8000
```

Para subir pelo Docker Compose, o README orienta `docker compose up --build api`; depois, a API fica na porta 8001 do host. A disponibilidade dos arquivos remotos e das versões de dependências deve ser considerada ao reproduzir o processo.

## 7. Glossário de engenharia

| Termo | Significado neste projeto |
|---|---|
| API / endpoint | Interface HTTP; `/predict` classifica e `/health` informa disponibilidade. |
| JSON | Formato de dados usado nos arquivos de origem e nos corpos da API. |
| CSV | Formato tabular usado para o dataset de treino e alguns relatórios. |
| ID | Identificador de proposição, usado para unir o texto aos temas. |
| Artefato | Arquivo produzido pelo treino e consumido em inferência; aqui, `.joblib`. |
| Serialização | Gravar um objeto para carregá-lo depois. `joblib` faz isso para o pipeline. |
| Ingestão | Buscar dados da fonte e disponibilizá-los localmente para processamento. |
| Inferência | Aplicar um modelo já treinado a uma entrada nova. |
| Tracking / MLflow | Registro de parâmetros e métricas de execuções de treino para comparação. |
| Worker | Processo servidor que atende requisições; o contêiner é configurado com três. |
| Contêiner | Ambiente empacotado com código e dependências para executar a aplicação. |
| `fit` | Ajustar/aprender os parâmetros do estimador a partir dos exemplos. |
| `predict_proba` | Estimar uma probabilidade ou score probabilístico para cada classe. |
| `argmax` | Encontrar a posição do maior valor em uma lista ou vetor. |

## 8. Sequência de leitura do código

Para entender o projeto sem começar por todas as bibliotecas, leia nesta ordem:

1. `README.md`: problema, comandos e limites declarados.
2. `app/main.py`: formato de entrada, validação e endpoints.
3. `app/inference.py`: como uma ementa vira resposta.
4. `app/dataset.py`: como exemplos e rótulos são montados.
5. `app/train.py`: como os dados viram um modelo e como ele é avaliado.
6. `app/baixar_dados.py`: origem e persistência dos dados.
7. `tests/`: exemplos executáveis do comportamento esperado.
8. `outputs/test_metrics.json` e `docs/analise-erros.md`: desempenho e falhas observadas.

Ao ler `app/train.py`, faça primeiro o rastreamento de uma única amostra: texto bruto → texto vetorizado → vetor de rótulos → estimativas por tema → métrica/resposta. Essa linha mental ajuda a ligar o código de ML ao ciclo de vida de software familiar.

## 9. Perguntas para checar entendimento

1. Por que os dados oficiais fazem este problema ser multilabel mesmo que a resposta da API contenha um único tema?
2. Qual diferença prática há entre selecionar todos os temas acima de 0,5 e selecionar apenas o maior score?
3. Para descobrir se o sistema está acrescentando temas indevidos, você olharia primeiro precisão ou recall? E para encontrar temas omitidos?
4. Por que manter ementas idênticas em uma única partição torna a avaliação mais confiável?
5. O que seria necessário para sustentar a afirmação de que o modelo generaliza para proposições de anos futuros?

## Respostas curtas

1. Uma proposição pode ter várias associações oficiais; o endpoint escolhe uma delas como saída resumida.
2. O limiar pode produzir nenhum ou vários temas; `argmax` produz um tema.
3. Precisão evidencia falsos positivos/temas extras; recall evidencia falsos negativos/temas omitidos.
4. Caso contrário, o modelo pode ser avaliado em texto que já viu (ou em duplicata), inflando o resultado.
5. Uma avaliação temporal em que anos anteriores sejam usados no treino e um ano futuro seja reservado para teste, além de dados e rótulos representativos.

