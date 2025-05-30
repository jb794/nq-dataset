import json
import subprocess

with open("singleqafulldataset.jsonl", "r") as f:
    dataset = [json.loads(line) for line in f]

output_path = "finalqaset.jsonl"

def call_ollama(prompt):
    result = subprocess.run(
        ["ollama", "run", "llama3"],  # llama2:13b-chat
        input=prompt,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()

def parse_questions(text):
    lines = text.split("\n")
    questions = []
    for line in lines:
        line = line.strip()
        if line and any(char.isdigit() for char in line[:3]):
            question_text = line.split(".", 1)[-1].strip()
            questions.append(question_text)
    return questions

with open(output_path, "w") as out_f:
    for i, qa in enumerate(dataset):
        original_question = qa["question"]
        answer = qa["answer"]

        prompt = f"""
You are given a question and an answer. The answer is correct and complete. Your task is to generate 9 additional questions that could also be answered by the same answer. These questions should be diverse — either rephrased versions of the original or questions that focus on different parts of the answer. Return only the 9 questions in a numbered list.

Original Question: {original_question}

Answer: {answer}
        """
        try:
            llama_output = call_ollama(prompt)
            generated_questions = parse_questions(llama_output)

            if len(generated_questions) != 9:
                print(f"Warning: Entry {i} returned {len(generated_questions)} questions")
                continue  

            all_questions = [original_question] + generated_questions

            final_entry = {
                "questions": all_questions,
                "answer": answer
            }

            out_f.write(json.dumps(final_entry) + "\n")
            print(f"Processed entry {i + 1}/{len(dataset)}")

        except Exception as e:
            print(f"Error processing entry {i}: {e}")
