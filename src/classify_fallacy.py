import pandas as pd
from openai import OpenAI
import os
import random
import csv
import argparse
import yaml
import json
from utils import metrics, classwise_eval, log_metrics_to_csv

parser = argparse.ArgumentParser(description="Logical fallacies classification")
parser.add_argument("--model", type=str, default="o4-mini", help="Model to use for classification.")
parser.add_argument("--prompt", type=str, default="zero-shot", help="Prompting method to use for classification.")
parser.add_argument("--dataset", type=str, default="logic", help="Dataset to classify.")

args = parser.parse_args()

def format_guidelines(guidelines):
    parts = []
    for item in guidelines:
        fallacy = item.get("fallacy", "Unknown Fallacy")
        guideline = item.get("guidelines", {})
        
        entry = f"{fallacy}:\n"
        entry += f"- Core definition: {guideline.get('core_definition', 'N/A')}\n"
        
        if "key_indicators" in guideline:
            entry += "- Key indicators:\n"
            entry += "".join(f"  - {ind}\n" for ind in guideline["key_indicators"])
        
        if "typical_confusion_patterns" in guideline:
            entry += "- Typical confusion patterns:\n"
            entry += "".join(f"  - {pat}\n" for pat in guideline["typical_confusion_patterns"])
        
        if "quick_checklist" in guideline:
            entry += "- Quick checklist:\n"
            entry += "".join(f"  - {chk}\n" for chk in guideline["quick_checklist"])
        
        parts.append(entry)
    
    return "\n".join(parts)


def get_prompt(user_prompt, fallacies, text, prompt=args.prompt, attr=None):
        if args.prompt == "zero-shot" or args.prompt == "exp":
            return user_prompt.format(fallacies=fallacies, text=text)
        else:
            return user_prompt.format(fallacies=fallacies, attr=attr, text=text)


def classify_fallacy(sentence, model, system_prompt, user_prompt):

        if args.model == "gpt4" or args.model == "o4-mini" or args.model == "gpt-4o":
            openai = OpenAI(
            api_key="x"
        )
            
        else:
            openai = OpenAI(
    api_key="x",
    base_url="https://api.deepinfra.com/v1/openai",
    )
            
        prompt = get_prompt(user_prompt, fallacies, prompt=args.prompt, attr=attr, text=sentence)

        print(prompt)
        chat_completion = openai.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                     {"role": "user", "content": prompt}],
            stream=False
        )
        return chat_completion.choices[0].message.content

if args.model == "o4-mini":
    model = "o4-mini"
elif args.model == "gpt4":
    model = "gpt-4.1-mini"
elif args.model == "gpt-4o":
    model = "gpt-4o"
elif args.model == "gemma":
    model = "google/gemma-3-27b-it"
elif args.model == "llama":
    model = "meta-llama/Llama-3.3-70B-Instruct"
elif args.model == "deepseek":
    model = "deepseek-ai/DeepSeek-R1"


if args.dataset == "logic":
    test = pd.read_csv("../data/logic_test.tsv", sep = "\t")
    output_folder = "logic"
elif args.dataset == "elecdebate":
    test = pd.read_csv("../data/elecdebate_test.csv")
elif args.dataset == "reddit":
    test = pd.read_csv("../data/reddit_test.tsv", sep = "\t")

test_sentences = test["text"].tolist()
gold_labels = test["fallacy_label"].tolist()
fallacy_list = test["fallacy_label"].unique().tolist()
fallacies = (", ").join(fallacy_list)


with open("../prompts/prompts.yaml", "r") as file:
    prompts_file = yaml.safe_load(file)


system_prompt = prompts_file[args.prompt]["system_prompt"][0]
user_prompt = prompts_file[args.prompt]["user_prompt"]

attr = None

if args.prompt == "def":
    with open(f"../features/{args.dataset}/definitions.json", "r") as f:
        definitions = json.load(f)
    def_in_prompt = f"""\n"""
    def_in_prompt += ("\n").join(definitions)
    attr = def_in_prompt
        
if args.prompt == "logical_form":
    with open(f"../features/{args.dataset}/logical_forms.json", "r") as f:
        logical_forms = json.load(f)
    map_in_prompt = f"""\n"""
    for entry in logical_forms:
        map_in_prompt += f"{entry['fallacy']}: {entry['logical_form']}\n"
    attr = map_in_prompt

if args.prompt == "patterns" or args.prompt == "patterns_matching" or args.prompt == "multistep":
    with open(f"../features/{args.dataset}/logic_generated_patterns.json", "r") as f:
        patterns_list = json.load(f)
    patterns_str = f"""\n"""
    for pattern in patterns_list:
        patterns_str += f"{pattern['fallacy']}:\n"
        for i, pat in enumerate(pattern["patterns"],1):
            patterns_str += f"{i}. {pat}\n"
    attr = patterns_str

if args.prompt == "new_def":
    with open(f"../features/{args.dataset}/generated_definitions.json", "r") as f:
        summaries = json.load(f)
    def_in_prompt = f"""\n"""
    def_in_prompt += ("\n").join(summaries)
    attr = def_in_prompt

# if args.prompt == "guidelines":
#     with open(f"features/guidelines/{args.model}_guidelines.json", "r") as f:
#         guidelines = json.load(f)
#     formatted_guideline = format_guidelines(guidelines)
#     attr = formatted_guideline


csv_file = f"{args.model}/results/{args.dataset}/results_{args.model}_{args.prompt}.csv"

if not os.path.exists(csv_file):
    df_header = pd.DataFrame(columns=["text", "gold_label", "predicted_label"])
    df_header.to_csv(csv_file, index=False)

answers = []
for i, sentence in enumerate(test_sentences):
    answer = classify_fallacy(sentence, model, system_prompt, user_prompt)
    answers.append(answer)

    row = pd.DataFrame({
        "text": [sentence],
        "gold_label": [gold_labels[i]],
        "predicted_label": [answer]
    })
    row.to_csv(csv_file, mode='a', header=False, index=False)
    
    print(f"Processed {i+1}/{len(test_sentences)}")  
print(f"Results saved to {csv_file}")

accuracy, microf1, macrof1, precision, recall = metrics(answers, gold_labels, fallacy_list)
log_metrics_to_csv(f"{args.model}/results/{args.dataset}/metrics.csv", accuracy, microf1, macrof1, recall, precision, args.prompt)
classwise_eval(gold_labels, answers, f"{args.model}/classwise/{args.dataset}/{args.prompt}.csv", fallacy_list)
