import json
import subprocess
import re

input_path = "finalqaset.jsonl"
output_path = "expanded30qaset.jsonl"

with open(input_path, "r") as f:
    dataset = [json.loads(line) for line in f]

def call_ollama(prompt):
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

def generate_questions(existing_questions, answer):
    prompt = (
        f"You are a helpful assistant. Your task is to generate 20 **new** and **unique** questions "
        "that could reasonably have the same answer as the one provided. The output should follow Zipf’s Law — "
        "most questions should be increasingly rare, specific, or detailed.\n\n"
        "Do NOT repeat any of the original questions listed below.\n\n"
        f"Original questions:\n{json.dumps(existing_questions, indent=2)}\n\n"
        f"Answer: \"{answer}\"\n\n"
        f"Return exactly 20 new questions in numbered format (1 to 20). Do not include explanations or restate the answer."
    )

    output = call_ollama(prompt)
    lines = output.strip().split("\n")

    # Filter only numbered lines like "1. Question...", "2) Question...", etc.
    numbered_lines = [line.strip() for line in lines if re.match(r"^\s*\d+[\.\)]", line.strip())]

    # Remove leading numbers and whitespace
    new_questions = [re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip() for line in numbered_lines]

    # Filter out duplicates
    original_set = set(q.strip().lower() for q in existing_questions)
    final_questions = []
    for q in new_questions:
        if q.strip().lower() not in original_set:
            final_questions.append(q)
        if len(final_questions) == 20:
            break

    return final_questions

with open(output_path, "w") as out_f:
    for item in dataset:
        original_qs = item["questions"]
        answer = item["answer"]

        try:
            new_qs = generate_questions(original_qs, answer)
            full_qs = original_qs + new_qs

            if len(full_qs) < 30:
                continue

            out_entry = {
                "questions": full_qs,
                "answer": answer
            }

            out_f.write(json.dumps(out_entry) + "\n")
        except Exception:
            continue
