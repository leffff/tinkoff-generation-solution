# tinkoff-generation-solution
### Мое решение 6 задачи отбора на Тинькофф поколение

Описание модели ```Doc2VecLM```: </br>
Модель представвляет собой гибрид Word2Vec (Skipgram), TF-IDF и линейного классификатора

Описание работы модели ```Doc2VecLM```:
1) ```Word2VecWrapper``` возвращает эмбеддинги для каждого слова во входной последовательности
2) ```TFIDFWrapper``` возвращает tfidf скор каждого слова во входной последовательности
3) ```Word2VecWrapper``` эмбеддинги умножаются на ```TFIDFWrapper``` скоры и усредняются, что представляет собой общий контекст предложения
4) ```Classifier``` принимает на вход эмбеддинг контекста и эмбеддинги ```last_n``` последних слов

