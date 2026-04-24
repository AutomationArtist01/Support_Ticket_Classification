from flask import Flask, request, jsonify, render_template
import fasttext
import re
from datetime import datetime
from deep_translator import GoogleTranslator

app = Flask(__name__)

model = fasttext.load_model("model/fasttext_model.ftz")

LABELS = {
    "credit_reporting": "Credit Reporting",
    "debt_collection": "Debt Collection",
    "loans": "Loans",
    "bank_accounts_and_services": "Bank Accounts & Services",
    "credit_card_services": "Credit Card Services"
}

complaints = []
counter = 1

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'x+', '', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def translate_text(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

def predict(text):
    text = translate_text(text)
    text = clean_text(text)

    labels, probs = model.predict(text)

    label_key = labels[0].replace("__label__", "")
    confidence = round(probs[0] * 100, 2)

    return label_key, confidence


@app.route("/")
def home():
    return render_template("home.html", labels=LABELS)

@app.route("/jobs")
def jobs():
    return render_template("jobs.html", labels=LABELS)


@app.route("/api/submit", methods=["POST"])
def submit():
    global counter
    data = request.get_json()
    name = data["name"]
    account = data["account"]
    text = data["text"]
    label_key, confidence = predict(text)
    complaint = {
        "id": counter,
        "name": name,
        "account": account,
        "original_text": text,
        "label_key": label_key,
        "label_name": LABELS.get(label_key, label_key),
        "confidence": confidence,
        "status": "open",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

    complaints.append(complaint)
    counter += 1

    return jsonify({
        "complaint_id": complaint["id"],
        "label_key": label_key,
        "label_name": complaint["label_name"],
        "confidence": confidence
    })


@app.route("/api/complaints")
def get_complaints():
    label = request.args.get("label")

    if label:
        return jsonify([c for c in complaints if c["label_key"] == label])
    return jsonify(complaints)

@app.route("/api/resolve/<int:id>", methods=["POST"])
def resolve(id):
    for c in complaints:
        if c["id"] == id:
            c["status"] = "resolved"
            break
    return jsonify({"ok": True})

@app.route("/api/redirect/<int:id>", methods=["POST"])
def redirect_ticket(id):
    data = request.get_json()
    new_label = data["new_label"]

    for c in complaints:
        if c["id"] == id:
            c["label_key"] = new_label
            c["label_name"] = LABELS[new_label]
            c["status"] = "redirected"
            break

    return jsonify({
        "ok": True,
        "new_label_name": LABELS[new_label]
    })

@app.route("/history")
def history():
    return render_template("history.html", labels=LABELS)

@app.route("/api/history")
def get_history():
    return jsonify([c for c in complaints if c["status"] == "resolved"])

if __name__ == "__main__":
    app.run(debug=True)