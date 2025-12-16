# Базовый образ с Python
FROM python:3.10-slim

# Отключаем создание .pyc и буферизацию stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Рабочая директория внутри контейнера
WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код сервиса и конфигурацию
COPY collector.py .
COPY config.yaml .

# Создаём директории для данных и логов
RUN mkdir -p data user_data

# Открываем порт сервиса
EXPOSE 5000

# Команда запуска сервиса
CMD ["python", "collector.py"]