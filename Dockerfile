FROM python:3.13.16-bookworm

# Configurações do ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/srv/app \
    TZ=America/Sao_Paulo

WORKDIR /srv/app

# Instala dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tzdata \
        libpq5 && \
    rm -rf /var/lib/apt/lists/*

# Instala dependências Python
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação
COPY . .

# Cria usuário não privilegiado
RUN useradd --create-home --shell /usr/sbin/nologin appuser && \
    chown -R appuser:appuser /srv/app

USER appuser

# Porta do FastAPI
EXPOSE 8000

# Inicializa a aplicação FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "3"]