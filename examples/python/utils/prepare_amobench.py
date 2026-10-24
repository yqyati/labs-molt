"""Convert answer-verifiable AMO rows to the Molt science-agent schema."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=50)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
    if len(rows) != args.expected_rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Unexpected row count or duplicate AMO ids")
    converted = []
    for row in rows:
        question, answers = row["question"], row["answer"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing question: {row['id']}")
        if isinstance(answers, str):
            answers = [answers]
        if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
            raise ValueError(f"Missing reference answers: {row['id']}")
        reference = json.dumps(answers, ensure_ascii=False)
        item = dict(row, answer=reference, question_id=row["id"], source="amobench")
        converted.append(dict(id=row["id"], datasource="amobench",
                              prompt=[{"role": "user", "content": question}],
                              answer=reference, science_item=item))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in converted))
    print(f"Prepared {len(converted)} AMO rows: {args.output}")


if __name__ == "__main__":
    main()
