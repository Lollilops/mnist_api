# Базовый образ Python
FROM python:3.10-slim

# Переменные окружения
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Рабочая директория
WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код и конфигурацию
COPY web_master.py .
COPY config.yaml .

# Порт Web Master (API Gateway)
EXPOSE 8000

# Запуск сервиса
CMD ["python", "web_master.py"]