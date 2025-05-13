from flask import Flask, render_template, request, jsonify # type: ignore
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
import torch # type: ignore

# Load the trained model and tokenizer
model_path = "./ner_model"
model = DistilBertForTokenClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

label_list = [
    "O",        # Outside any entity
    "B-PER",    # Beginning of a Person name
    "I-PER",    # Inside of a Person name
    "B-ORG",    # Beginning of an Organization name
    "I-ORG",    # Inside of an Organization name
    "B-LOC",    # Beginning of a Location name
    "I-LOC",    # Inside of a Location name
    "B-MISC",   # Beginning of a Miscellaneous entity
    "I-MISC"    # Inside of a Miscellaneous entity
]



app = Flask(__name__)

def predict_entities(text):
    tokens = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    model.eval()

    with torch.no_grad():
        outputs = model(**tokens)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predictions = torch.argmax(probs, dim=2)
    confidence_scores = torch.max(probs, dim=2).values

    input_ids = tokens["input_ids"][0]
    tokenized_text = tokenizer.convert_ids_to_tokens(input_ids)
    predicted_labels = predictions[0].tolist()
    confidence_values = confidence_scores[0].tolist()

    formatted_output = []
    entity, entity_label, entity_confidence = "", "", 0.0

    for token, label_id, confidence, input_id in zip(tokenized_text, predicted_labels, confidence_values, input_ids):
        # Skip special tokens like [PAD], [CLS], [SEP]
        if token in tokenizer.all_special_tokens:
            continue

        label = label_list[label_id] if label_id < len(label_list) else "O"

        if label.startswith("B-"):
            if entity:
                formatted_output.append((entity.strip(), entity_label, entity_confidence))
            entity = token.replace("##", "")
            entity_label = label[2:]
            entity_confidence = confidence

        elif label.startswith("I-") and entity:
            entity += " " + token.replace("##", "")
            entity_confidence = min(entity_confidence, confidence)

        else:
            if entity:
                formatted_output.append((entity.strip(), entity_label, entity_confidence))
                entity, entity_label, entity_confidence = "", "", 0.0

    if entity:
        formatted_output.append((entity.strip(), entity_label, entity_confidence))

    return formatted_output


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        entities = predict_entities(text)
        return jsonify(entities)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
