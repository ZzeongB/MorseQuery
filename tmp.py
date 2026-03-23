import json
import re

text = open("./raw_text.txt", "r", encoding="utf-8").read()

# improved pattern: stop D) at next question start (capital letter paragraph) not newline
pattern = re.compile(
    r"(?P<stem>.*?)(?:\n|\r\n)\s*A\)\s*(?P<A>.*?)\n\s*B\)\s*(?P<B>.*?)\n\s*C\)\s*(?P<C>.*?)\n\s*D\)\s*(?P<D>.*?)(?=\n[A-Z]|$)",
    re.S,
)

questions = []
for m in pattern.finditer(text):
    stem = m.group("stem").strip()
    stem = re.sub(r"Which choice.*?\?\s*$", "", stem, flags=re.S).strip()

    options = [
        {"text": m.group("A").strip()},
        {"text": m.group("B").strip()},
        {"text": m.group("C").strip()},
        {"text": m.group("D").strip()},
    ]
    questions.append({"text": stem, "options": options})

quiz = {"title": "SAT Quiz", "description": "Improved parsing", "questions": questions}

out_path = "./sat_quiz.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(quiz, f, indent=2, ensure_ascii=False)

len(questions), out_path
