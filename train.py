import fasttext
import os

if not os.path.exists("data/train.txt"):
    print("train.txt not found")
    exit()

if not os.path.exists("data/test.txt"):
    print("test.txt not found")
    exit()


model = fasttext.train_supervised(
    input="data/train.txt",
    autotuneValidationFile="data/test.txt",
    autotuneDuration=600,
    autotuneMetric="f1",
    wordNgrams=3,
    loss="softmax"
)

os.makedirs("model", exist_ok=True)
model.save_model("model/fasttext_model.ftz")

result = model.test("data/test.txt")

print("Total samples:", result[0])
print("Precision:", round(result[1], 3))
print("Recall:", round(result[2], 3))