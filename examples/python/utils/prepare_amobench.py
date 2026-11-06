"""Convert AMO, IMO-AnswerBench or CMT rows to the Molt science-agent schema."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=50)
    parser.add_argument("--benchmark", choices=["amobench", "imo_answerbench", "cmt_benchmark"], default="amobench")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
    if len(rows) != args.expected_rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Unexpected row count or duplicate benchmark ids")
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
        prompt = [{"role": "user", "content": question}]
        if args.benchmark == "imo_answerbench":
            prompt = row["prompt"]
            if len(prompt) != 1 or prompt[0].get("role") != "user" or not prompt[0].get("content", "").strip():
                raise ValueError(f"Invalid IMO prompt: {row['id']}")
        elif args.benchmark == "cmt_benchmark":
            if row.get("image_question"):
                raise ValueError(f"Image input requires an explicit adapter: {row['id']}")
            prompt[0]["content"] = (
                "Use LaTeX for formulas. Show your reasoning, then write FINAL ANSWER: "
                "followed by the final answer(s) in \\boxed{}. "
                "For multiple answers, preserve their order and use separate boxes. "
                "Do not use boxes in the reasoning.\n\n"
                f"Problem:\n{question}\n\nParameters (if any):\n{row.get('parameters') or 'N/A'}"
                f"\n\nFunctions (if any):\n{row.get('functions') or 'N/A'}\n"
            )
        item = dict(row, answer=reference, question_id=row["id"], source=args.benchmark)
        converted.append(dict(id=row["id"], datasource=args.benchmark,
                              prompt=prompt,
                              answer=reference, science_item=item))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in converted))
    print(f"Prepared {len(converted)} {args.benchmark} rows: {args.output}")


if __name__ == "__main__":
    main()
