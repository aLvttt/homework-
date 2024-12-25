from gensim.models import KeyedVectors
from gensim.models import FastText
from nltk.tokenize import word_tokenize
import numpy as np

'''
засчитайте пж классную работу, у меня возникла проблема с тем, что нужна старая версия gensim,
а у меня корректно устанавливается только новая. Давайте сделаем вид, что я впитал весь материал :,(


P.S.: архив модели находится на гугл диске по ссылке: https://drive.google.com/file/d/1eM9mt6agwzIXHHa9OVct04cBWlEqLIOR/view?usp=sharing
(в коммит не поместились)

'''



model_path = "classwork_5\datasettt\model.model"

try:
    w2v_model = FastText.load(model_path)
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    exit()

texts = [
    "У берегов Коста-Рики водятся 1600 видов рыб. А на берег выходят черепахи весом до 500 кг. Но я не видел :(",
    "Нету рыб, уже холодно :( какую-то фигню зажгла, вроде потеплее :(",
    "Я так мечтала о тортике и колбаске, но бабушка жестоко продинамила меня, приготовив пирожки и рыбу",
    "Цитрусовые, яйца, клубнику, малину, дыни, кофе, помидоры, мед, молоко, копченые изделия, манную кашу, рыбу, шоколад. Нельзя, как я буду жить без этого"
]

def tokenize(text):
    return word_tokenize(text.lower())

def get_text_vector(text, model):
    tokens = tokenize(text)
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

text_vectors = [get_text_vector(text, w2v_model) for text in texts]

from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(text_vectors)

print("Сходство между текстами:")
print(similarity_matrix)

print("\nПример вектора для первого текста:")
print(text_vectors[0])