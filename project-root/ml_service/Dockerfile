# Базовый образ
FROM python:3.11

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

# Копируем код
COPY ml_service.py .
COPY model.py .
COPY config.yaml .

# Директории для моделей и данных
RUN mkdir -p models data
COPY ./models ./models

# Порт сервиса
EXPOSE 5000

# Запуск
CMD ["python", "ml_service.py"]