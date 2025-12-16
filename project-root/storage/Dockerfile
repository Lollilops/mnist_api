# Базовый образ Python
FROM python:3.10-slim

# Переменные окружения
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Рабочая директория
WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код и конфигурацию
COPY storage.py .
COPY config.yaml .

# Создаём директорию для данных хранилища
RUN mkdir -p storage_data/images storage_data/models

# Открываем порт
EXPOSE 5000

# Запуск сервиса
CMD ["python", "storage.py"]