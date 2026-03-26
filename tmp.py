import json
import re

text = open("./raw_text.txt", "r", encoding="utf-8").read()


def clean_text(x):
    return x
    # return re.sub(r"\s*\n\s*", " ", x).strip()


# 엔터 여러 개 기준으로 block 분리
blocks = [b.strip() for b in re.split(r"(?:\r?\n){2,}", text) if b.strip()]

questions = []

for block in blocks:
    # 보기 4개가 모두 있는 block만 문제로 간주
    m = re.search(
        r"(?s)^(?P<stem>.*?)\s*A\)\s*(?P<A>.*?)\s*B\)\s*(?P<B>.*?)\s*C\)\s*(?P<C>.*?)\s*D\)\s*(?P<D>.*?)\s*$",
        block,
    )
    if not m:
        continue

    stem = clean_text(m.group("stem"))

    options = [
        {"text": clean_text(m.group("A"))},
        {"text": clean_text(m.group("B"))},
        {"text": clean_text(m.group("C"))},
        {"text": clean_text(m.group("D"))},
    ]

    questions.append({"text": stem, "options": options})

quiz = {"title": "PSAT Quiz", "description": "", "questions": questions}

out_path = "./sat_quiz.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(quiz, f, indent=2, ensure_ascii=False)

print(len(questions), out_path)
