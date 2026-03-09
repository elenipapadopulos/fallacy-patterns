import pandas as pd
from openai import OpenAI
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
import random
import yaml
import json
import csv
import argparse
from utils import metrics, classwise_eval, log_metrics_to_csv

parser = argparse.ArgumentParser(description="Few-shot classification of logical fallacies.")
parser.add_argument("--model", type=str, default="o4-mini", help="Model to use for classification.")
parser.add_argument("--prompt", type=str, default="list", help="Prompting method to use for classification.")
parser.add_argument("--dataset", type=str, default="logic", help="Dataset to classify.")

args = parser.parse_args()


def format_patterns(patterns_list):
    return "\n".join(f"{i+1}. {pattern}" for i, pattern in enumerate(patterns_list))

def get_shuffled_examples():
    shuffled = examples.copy()
    random.shuffle(shuffled)
    return shuffled


def gather_examples(text, demo_dict):
    # the key of the dictionary is the sentence 
    retrieved_examples = demo_dict[text]
    return retrieved_examples

def get_prompt_dict(text, demo_dict):
    examples = gather_examples(text, demo_dict)
    examples = examples.copy()
    random.shuffle(examples)

    prefix = user_prompt
    suffix = f"""Now classify the following argument.
Argument: {text}
Fallacy:"""
    
    # format the 'patterns' field of each example before passing them to the prompt
    for ex in examples:
        if isinstance(ex.get("patterns"), list):
            ex["patterns"] = format_patterns(ex["patterns"])

    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["text"]
    )
    return prompt.format(text=text)

def classify_fallacy(sentence, model, examples, system_prompt):

        if args.model == "gpt4" or args.model == "o4-mini":
            openai = OpenAI(
            api_key="x"
        )
            
        else:
            openai = OpenAI(
    api_key="x",
    base_url="https://api.deepinfra.com/v1/openai",
    )
            
        user_prompt = get_prompt_dict(sentence, examples)

        print(user_prompt)

        chat_completion = openai.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]    
        )
        return chat_completion.choices[0].message.content

if args.model == "o4-mini":
    model = "o4-mini"
elif args.model == "gpt4":
    model = "gpt-4.1-mini"
elif args.model == "gemma":
    model = "google/gemma-3-27b-it"
elif args.model == "llama":
    model = "meta-llama/Llama-3.3-70B-Instruct"

if args.dataset == "logic":
    test = pd.read_csv("../data/logic_test.tsv", sep = "\t")
elif args.dataset == "elecdebate":
    test = pd.read_csv("../data/elecdebate_test.csv")
elif args.dataset == "reddit":
    test = pd.read_csv("../data/reddit_test.tsv", sep = "\t")

test_sentences = test["text"].tolist()
gold_labels = test["fallacy_label"].tolist()
fallacy_list = test["fallacy_label"].unique().tolist()
fallacies = (", ").join(fallacy_list)


with open("../prompts/prompts_one-shot.yaml", "r") as file:
    prompts_file = yaml.safe_load(file)

one_shot_prompts = ("one-shot") 
one_shot_prompts_exp = ("one-shot-exp", "one-shot-exp-patterns") 

if args.prompt in one_shot_prompts:
    system_prompt = prompts_file["one-shot"]["system_prompt"][0]
    user_prompt = prompts_file["one-shot"]["user_prompt"].format(fallacies=fallacies)

    example_prompt = PromptTemplate(
    input_variables=["text", "label"],
    template="Argument: {text}\nFallacy: {label}"
    )

elif args.prompt in one_shot_prompts_exp:

    system_prompt = prompts_file["one-shot-exp"]["system_prompt"][0]
    user_prompt = prompts_file["one-shot-exp"]["user_prompt"].format(fallacies=fallacies)

    if args.prompt == "one-shot-exp-patterns":
        example_prompt = PromptTemplate(
            input_variables=["text", "label", "explanation", "patterns"],
            template="Argument: {text}\nFallacy: {label}\nExplanation: {explanation}\n{label}'s patterns:\n{patterns}"
        )
    else:
        example_prompt = PromptTemplate(
        input_variables=["text", "label", "explanation"],
        template="Argument: {text}\nFallacy: {label}\nExplanation: {explanation}"
    )


with open(f"../features/{args.dataset}/dynamic_examples.json", "r") as f:
        examples = json.load(f)


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
log_metrics_to_csv(f"{args.dataset}/{args.model}/results/metrics.csv", gold_labels, answers, accuracy, microf1, macrof1, recall, precision, args.prompt)
classwise_eval(gold_labels, answers, f"{args.dataset}/{args.model}/classwise/{args.dataset}_{args.prompt}.csv", fallacy_list)