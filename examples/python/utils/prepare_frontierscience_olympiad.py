"""Convert FrontierScience Olympiad to the Molt science-agent evaluation schema."""
import argparse
import json
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=100)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
    if len(rows) != args.expected_rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Unexpected row count or duplicate FSO ids")
    converted = []
    for row in rows:
        question, answer = row["question"], row["answer"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing question: {row['id']}")
        if isinstance(answer, list):
            if not answer or any(not isinstance(a, str) or not a.strip() for a in answer):
                raise ValueError(f"Missing reference answer: {row['id']}")
            answer = "\n".join(answer)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Missing reference answer: {row['id']}")
        prompt = (
            question + "\n\nUse LaTeX for formulas. Put the final answer(s) inside <answer></answer>, "
            "with each requested answer in a separate \\boxed{...}, in the order requested. "
            "Keep labels and units clear."
        )
        item = dict(row, answer=answer, question_id=row["id"], source="frontierscience_olympiad")
        converted.append(dict(id=row["id"], datasource="frontierscience_olympiad",
                              prompt=[{"role": "user", "content": prompt}],
                              answer=answer, science_item=item))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in converted))
    print(f"Prepared {len(converted)} FSO rows: {args.output}")
