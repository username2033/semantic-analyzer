import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from collections import Counter
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import sklearn.metrics.pairwise
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np


import json

def read_json_data(filepath):

  try:
    with open(filepath, 'r', encoding='utf-8') as f: # Указание кодировки для корректного чтения
      data = json.load(f)
      if isinstance(data, list): # Проверка, является ли загруженная структура списком
        for item in data:
          if not all(key in item for key in ["question", "correct_answer", "students_answers", "number_of_columns"]):
            print("Ошибка: в JSON-файле отсутствуют необходимые ключи в одном или нескольких элементах.")
            return None
        return data
      else:
        print("Ошибка: JSON-файл не содержит списка словарей.")
        return None

  except FileNotFoundError:
    print(f"Ошибка: файл {filepath} не найден.")
    return None
  except json.JSONDecodeError:
    print(f"Ошибка: неверный формат JSON-файла {filepath}.")
    return None


inf = float('inf')
# Загружаем модель и токенайзер
model_name = "DeepPavlov/bert-base-cased-conversational"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# print(tokenizer.model_max_length)
model = AutoModel.from_pretrained(model_name)
stop_words = set(stopwords.words('russian'))
lemmatizer = WordNetLemmatizer()

def lower(arr):
    result = ["" for i in range(len(arr))]
    for i in range(len(arr)):
        result[i] = arr[i].lower()
    return result[:]

def key_words(text):
    global stop_words, lemmatizer
    #tokens = word_tokenize(text)
    key_words = set([lemmatizer.lemmatize(w) for w in word_tokenize(text) if w.isalnum() and w not in stop_words])
    if len(key_words) > 0:
        return key_words
    return set(map(str, text.split()))
def key_words_list(text):
    global stop_words, lemmatizer
    #tokens = word_tokenize(text)
    key_words = list([lemmatizer.lemmatize(w) for w in word_tokenize(text) if w.isalnum() and w not in stop_words])
    if len(key_words) > 0:
        return key_words
    return list(map(str, text.split()))


def preprocess_and_analyze_answers(answers, correct_answer_words):
    global stop_words
    #global correct_answer_words
    lemmatizer = WordNetLemmatizer()

    all_tokens = []
    answer_correctness = []

    for answer in answers:
        # Токенизация
        # Удаление стоп-слов и лемматизация
        filtered_tokens = key_words_list(answer.lower())
        all_tokens.append(filtered_tokens)

        # Примитивная проверка правильности (на основе совпадения ключевых фраз)
        correctness = 0
        counter = 0
        filtered_tokens_set = set(filtered_tokens)
        for word in correct_answer_words:
            if word in filtered_tokens_set:
                counter += 1
        correctness = counter / len(correct_answer_words)
        #correctness = set(filtered_tokens) & correct_answer_words
        answer_correctness.append(correctness)

    return all_tokens, answer_correctness

def get_embedding(text):
  # Токенизация текста
  inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
  print(inputs, "tokenizer")
  #print(inputs, text)
  with torch.no_grad():
    # Получение выходных данных из BERT
    outputs = model(**inputs)
  # Вычисляем средний слой токенов для векторного представления текста
  embeddings = outputs.last_hidden_state.mean(dim=1)
  return embeddings

def getDist(correct, answers):
  correct_emb = get_embedding(correct)
  results = []

  for answer in answers:
    answer_emb = get_embedding(answer)
    similarity = sklearn.metrics.pairwise.euclidean_distances(correct_emb, answer_emb)
    results.append(similarity.flatten()[0])

  return results

def is_right(distances, correctness):
    rightAnswers = []
    result = [False for i in range(len(distances))]
    #measure = [0 for i in range(len(distances))]
    measure = {}
    for i in range(len(distances)):
        #print(distances[i] / correctness[i], student_answers[i])
        if distances[i] == 0 and correctness[i] != 0:
            rightAnswers.append(i)
            result[i] = True
            measure[i] = 0
        elif correctness[i] != 0:
            # print(distances[i] / correctness[i], student_answers[i])
            measure[i] = distances[i] / correctness[i]
            if measure[i] < 9:
                rightAnswers.append(i)
                result[i] = True

    return rightAnswers, result, measure
# Данные
#correct_answer = "Глобальное потепление - это увеличение средней температуры атмосферы и океанов Земли."
def analize(correct_answer, student_answers):
    correct_answer_words = key_words(correct_answer.lower())
    print(correct_answer_words)
    tokens, correctness = preprocess_and_analyze_answers(student_answers, correct_answer_words)
    # Проверка
    distances = getDist(correct_answer, student_answers)
    rightAnswers, similarities, measure = is_right(distances, correctness)
    #print(measure, "measure")
    #plt = create_histogram([measure[i] for i in measure], 4)
    return similarities, correctness, distances, tokens, rightAnswers, measure, correct_answer_words

def analize_question(task):
    #{"question": question, "correct_answer": correct_answer, "students_answers": students_answers}
    return analize(task["correct_answer"], task["students_answers"])

def create_histogram(arr, n):
    if len(arr) > 0:
        #print(arr, "arr")
        min_val = min(arr)
        max_val = max(arr)
        h = (max_val - min_val) / n if max_val != min_val else 1 #Обработка случая, когда все значения одинаковы


        counts = [0] * n
        for x in arr:
            bin_index = int((x - min_val) // h)
            if bin_index == n: #корректировка для случаев, когда x == max_val
                bin_index -=1
            counts[bin_index] += 1
    else:
        min_val = 0
        h = 1

    bins = [min_val + i * h for i in range(n + 1)]  # Границы интервалов
    plt.hist(arr, bins=bins, edgecolor='black')
    plt.xlabel("Значения")
    plt.ylabel("Частота")
    plt.title(f"Гистограмма из {n} столбцов")
    plt.xticks(bins) #Устанавливаем отметки на оси Х в соответствии с границами интервалов.
    # plt.show()
    return plt


class QuizApp:
    def __init__(self, root):
        self.root = root
        #self.number_of_columns = 4
        self.data = [] # Здесь будем хранить вопросы и ответы
        root.title("Сбор данных вопросов")

        # Окно ввода данных
        self.question_label = tk.Label(root, text="Вопрос:")
        self.question_label.grid(row=0, column=0)
        self.question_entry = tk.Entry(root, width=120)
        self.question_entry.grid(row=0, column=1)

        self.answer_label = tk.Label(root, text="Правильный ответ:")
        self.answer_label.grid(row=1, column=0)
        self.answer_entry = tk.Entry(root, width=120)
        self.answer_entry.grid(row=1, column=1)

        self.students_label = tk.Label(root, text="Ответы учеников (через точку с запятой):")
        self.students_label.grid(row=2, column=0)
        self.students_entry = tk.Entry(root, width=120)
        self.students_entry.grid(row=2, column=1)

        self.number_of_columns = tk.Label(root, text="Количество столбцов в гистограммах (по умолчанию - 4):")
        self.number_of_columns.grid(row=3, column=0)
        self.number_of_columns = tk.Entry(root, width=35)
        self.number_of_columns.grid(row=3, column=1)

        self.save_button = tk.Button(root, text="Сохранить данные из окна", command=self.save_data)
        self.save_button.grid(row=4, column=0, columnspan=2)

        self.filepath = tk.Label(root, text="Путь до файла:")
        self.filepath.grid(row=5, column=0)
        self.filepath = tk.Entry(root, width=120)
        self.filepath.grid(row=5, column=1)

        self.get_from_file_button = tk.Button(root, text="Получить данные из файла (по умолчанию - 'questions.json')", command=self.get_data_from_json)
        self.get_from_file_button.grid(row=6, column=0, columnspan=2)

        self.analyze_button = tk.Button(root, text="Показать аналитику", command=self.show_analysis)
        self.analyze_button.grid(row=7, column=0, columnspan=2)

        self.delete_button = tk.Button(root, text="Задать количество столбцов для всех гистограмм", command=self.set_columns_num)
        self.delete_button.grid(row=8, column=0, columnspan=2)

        self.delete_button = tk.Button(root, text="Очистить данные", command=self.delete_data)
        self.delete_button.grid(row=9, column=0, columnspan=2)

    def set_columns_num(self):
        number_of_columns = self.number_of_columns.get()
        if len(number_of_columns) == 0:
            messagebox.showerror("Ошибка", "Задайте количество столбцов для всех гистограмм")
            return
        number_of_columns = int(number_of_columns)
        for id in range(len(self.data)):
            self.data[id]["number_of_columns"] = number_of_columns

    def delete_data(self):
        self.data = []

    def get_data_from_json(self):
        filepath = self.filepath.get()
        if len(filepath) == 0:
            filepath = r'questions.json'
        data = read_json_data(filepath)
        for question in data:
            self.data.append(question)
        messagebox.showinfo("Данные из файла сохранены", "Данные из файла сохранены")
        #messagebox.showerror("Данные из файла сохранены", "Данные из файла сохранены")
        print(f"Данные сохранены: {self.data}")
    # filepath = r'C:\Users\Тагир\questions1.json' # Замените на ваш путь к файлу
    # data = read_json_data(filepath)

    def save_data(self):
        question = self.question_entry.get()
        correct_answer = self.answer_entry.get()
        students_answers = self.students_entry.get().split('; ')
        number_of_columns = self.number_of_columns.get()
        if len(number_of_columns) == 0:
            number_of_columns = 4
        else:
            number_of_columns = int(number_of_columns)
        self.data.append({"question": question, "correct_answer": correct_answer, "students_answers": students_answers, "number_of_columns": number_of_columns})
        print(f"Данные сохранены: {self.data}")

    def create_histogram(self, parent_window, arr, n):
          if len(arr) > 0:
              min_val = min(arr)
              max_val = max(arr)
              h = (max_val - min_val) / n if max_val != min_val else 1
              counts = [0] * n
              for x in arr:
                  bin_index = int((x - min_val) // h)
                  if bin_index == n:
                      bin_index -= 1
                  counts[bin_index] += 1
          else:
              min_val = 0
              h = 1

          bins = [min_val + i * h for i in range(n + 1)]

          fig, ax = plt.subplots()
          ax.hist(arr, bins=bins, edgecolor='black')
          ax.set_xlabel("Значения")
          ax.set_ylabel("Частота")
          ax.set_title(f"Гистограмма из {n} столбцов")
          ax.set_xticks(bins)

          canvas = FigureCanvasTkAgg(fig, master=parent_window)
          canvas.draw()
          canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def show_analysis(self):
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("Аналитика")

        # Создаем Canvas и Scrollbar
        canvas = tk.Canvas(analysis_window)
        scrollbar_y = tk.Scrollbar(analysis_window, orient="vertical", command=canvas.yview)
        scrollbar_x = tk.Scrollbar(analysis_window, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        # Расположение scrollbar
        scrollbar_y.pack(side="right", fill="y")
        scrollbar_x.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)

        # Создаем Frame внутри Canvas
        frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=frame, anchor="nw")

        # Функция для обновления scrollregion
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        frame.bind("<Configure>", on_frame_configure)

        # Заполнение данными
        for entry in self.data:

            # Получение результатов анализа для каждого элемента
            similarities, correctness, distances, tokens, rightAnswers, measure, correct_answer_words = analize_question(entry)
            print(tokens)
            # {"question": question, "correct_answer": correct_answer, "students_answers": students_answers}
            text = "Вопрос: " + str(entry["question"])
            tk.Label(frame, text=text).pack(anchor='w')

            text1 = "Правильный ответ: " + str(entry["correct_answer"])
            tk.Label(frame, text=text1).pack(anchor='w')
            text1 = "Встречавшиеся в правильном ответе ключевые слова: " + str(correct_answer_words)
            tk.Label(frame, text=text1).pack(anchor='w')
            tk.Label(frame, text=" ").pack(anchor='w')

            for id in range(len(entry["students_answers"])):
                if id in measure:
                    text1 = 'Ответ "' + str(entry["students_answers"][id]) + '": Семантическое расстояние до верного ответа: ' + str(measure[id]) + ", Семантическое (Евклидово) расстояние до верного ответа: " + str(distances[id])
                else:
                    text1 = 'Ответ "' + str(
                        entry["students_answers"][id]) + '": Семантическое расстояние до верного ответа: ' + "inf" + ", Семантическое (Евклидово) расстояние до верного ответа: " + str(distances[id])
                tk.Label(frame, text=text1).pack(anchor='w')
                text1 = "Встречавшиеся в ответе ключевые слова: " + str(tokens[id])
                tk.Label(frame, text=text1).pack(anchor='w')

            # Создание гистограмм
            tk.Label(frame, text="Гистограмма семантических расстояний до верного ответа ").pack(anchor='w')
            self.create_histogram(frame, [measure[i] for i in measure], entry["number_of_columns"])

            tk.Label(frame, text="Гистограмма семантических расстояний до верного ответа по Евклидовой метрике").pack(anchor='w')
            self.create_histogram(frame, distances, entry["number_of_columns"])


def main():
  root = tk.Tk()
  app = QuizApp(root)
  root.mainloop()

if __name__ == "__main__":
  main()
