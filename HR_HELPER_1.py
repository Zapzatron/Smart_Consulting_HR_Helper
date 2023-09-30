from datetime import datetime as dt
import pytz as ptz
import numpy as np
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import telebot
import config
import os
import io
import traceback
import signal


class ExceptionHandler(telebot.ExceptionHandler):
    def handle(self, exception):
        string_manager = io.StringIO()
        traceback.print_exc(file=string_manager)
        error = string_manager.getvalue()
        print(f"\033[31m{'-' * 120}\n{error}\n{'-' * 120}\033[0m")
        return True


bot = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN, exception_handler=ExceptionHandler())

bot.set_my_commands([
    telebot.types.BotCommand("/start", "Перезапуск чата"),
    telebot.types.BotCommand("/help", "Документация"),
])


def get_time(tz: str | None = 'Europe/Moscow', form: str = '%d-%m-%Y %H:%M:%S', strp: bool = False):
    if strp:
        if tz:
            return dt.strptime(dt.now(ptz.timezone(tz)).strftime(form), form)
        else:
            return dt.strptime(dt.now().strftime(form), form)
    else:
        if tz:
            return dt.now(ptz.timezone(tz)).strftime(form)
        else:
            return dt.now().strftime(form)


def find_need_context(question, documents):
    # Преобразование текстов в TF-IDF векторы
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    # Преобразование вопроса в TF-IDF вектор
    question_vector = tfidf_vectorizer.transform([question])

    # Вычисление косинусного сходства между вопросом и каждым документом
    cosine_similarities = cosine_similarity(question_vector, tfidf_matrix)

    # Нахождение индекса документа с наибольшим сходством
    best_match_index = np.argmax(cosine_similarities)

    # Вывод наилучшего ответа
    best_match_document = documents[best_match_index]
    # print(f"Наилучший ответ в файле под номером {best_match_index}:")
    # print(best_match_document)

    return best_match_document


qa_model = pipeline("question-answering", model="AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru")


def get_answer_local(question, context):
    output = qa_model(question=question, context=context)["answer"]
    return output.strip()


def hr_helper(question):
    # Список файлов md
    file_names = ["code.md", "hr.md", "policies.md", "smart_guide.md"]
    documents = []

    for file_name in file_names:
        with open(file_name, 'r', encoding='utf-8') as file:
            documents.append(file.read())

    # print(question)
    # print(documents)

    best_match_document = find_need_context(question, documents)
    # print(best_match_document)
    output = get_answer_local(question, best_match_document)

    if output:
        return output
    else:
        return "Я тебя не понял, задай этот вопрос к HR напрямую."


@bot.message_handler(commands=["start", "help"])
def commands(message):
    user_id = message.from_user.id
    first_name = message.from_user.first_name
    last_name = message.from_user.last_name
    print(f"[{get_time()}] Id: {user_id} Fn: {first_name} Ln: {last_name} Do: {message.text}", flush=True)
    help_text = """
Привет, я HR помощник. Постараюсь ответить на твои вопросы.
1. Отправь свой вопрос в чат
2. Дождись ответа
"""
    bot.reply_to(message, help_text)


@bot.message_handler(content_types=["text"])
def hr_helper_bot(message):
    user_id = message.from_user.id
    first_name = message.from_user.first_name
    last_name = message.from_user.last_name
    print(f"[{get_time()}] Id: {user_id} Fn: {first_name} Ln: {last_name}", flush=True)
    bot.reply_to(message, "Вопрос отправлен на обработку...")
    question = message.text.strip()
    # print(text)

    start = get_time(strp=True)
    text = hr_helper(question).strip()
    end = get_time(strp=True)
    print("\n\n")
    print(f"{start} --> {end} = {end - start}", flush=True)
    print(f"Вопрос:\n{question}")
    print(f"Ответ на вопрос:\n{text}")

    # 2023-09-30 15:06:41 --> 2023-09-30 15:12:31 = 0:05:50
    # Вопрос:
    # Что такое Смартократия?
    # Ответ на вопрос:
    # система управления твоей компанией.

    bot.reply_to(message, text)


if __name__ == "__main__":
    try:
        print(f"[{get_time()}] Бот включён :)", flush=True)
        bot.polling()
    except KeyboardInterrupt:
        print(f"[{get_time()}] Бот выключен :(", flush=True)
        os.kill(os.getpid(), signal.SIGTERM)
