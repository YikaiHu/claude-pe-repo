import json
import boto3
import pandas as pd

from datetime import datetime

current_time = datetime.now().strftime("%Y-%m-%d-%H%M%S")

bedrock_runtime_client = boto3.client("bedrock-runtime")

verbose = 0

explanation_template = """请帮我用白话文解释产品{prod_catalog}的广告词 {source_text}

Please enclose the explanation in <explanation></explanation> XML tags.
"""

translate_template = """The slogan to be translated will be give in the xml tags: <paragraph></paragraph>
The translate result should be wrapped in xml tags: <translation></translation>

Instructions:
- 参考explanation里面的解释, 不要丢失 explanation 的内容
- 最大可能的保证信雅达
- 使用地道的目标语言
- 不要改变原文的语序
- 不要改变原文的目的
- 保留原文的修辞手法，例如夸张、比喻、拟人和双关

<example>
source_text: 易如反掌
destination_lang: English
translation: a piece of cake
</example>

The paragraph explanation is <explanation>
{explanation}
</explanation>

This is a slogan of {prod_catalog}.
请遵循上面的Instructions, 把下面的广告词改写成地道的 {destination_lang}, 你可以参考<example></example>中的例子
<paragraph>{source_text}</paragraph>
"""


def invoke_claude_msg(system, messages, model_size, task_type="translation"):
    if verbose == 2:
        print(f"[SP]: {system}\n[messages]: {messages}")
    elif verbose == 1:
        print(f"[messages]: {messages}")

    if model_size == "haiku":
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    elif model_size == "sonnet":
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    else:
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

    if task_type == "translation":
        claude3_params = {
            "body": json.dumps(
                {
                    "system": system,
                    "messages": messages,
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 20000,
                    "stop_sequences": ["</translation>"],
                    "top_p": 0.5,
                    "top_k": 50,
                    "temperature": 0.1,
                }
            ),
            "modelId": model_id,
        }
    elif task_type == "explanation":
        claude3_params = {
            "body": json.dumps(
                {
                    "system": system,
                    "messages": messages,
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 20000,
                    "stop_sequences": ["</explanation>"],
                    "top_p": 0.5,
                    "top_k": 50,
                    "temperature": 1.0,
                }
            ),
            "modelId": model_id,
        }
    else:
        claude3_params = {
            "body": json.dumps(
                {
                    "system": system,
                    "messages": messages,
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 20000,
                    "stop_sequences": ["</translation>"],
                    "top_p": 0.5,
                    "top_k": 50,
                    "temperature": 0.1,
                }
            ),
            "modelId": model_id,
        }

    response = bedrock_runtime_client.invoke_model(**claude3_params)
    body = json.loads(response["body"].read().decode())
    return body["content"][0]["text"]


def process_excel(input_file_path, output_file_path, destination_lang, model_size):
    df = pd.read_excel(input_file_path)

    system_prompt = ""

    result_df = pd.DataFrame(columns=["source", "think", "destination"])

    for index, row in df.iterrows():
        source_text = replace_punctuation(row["source"])
        prod_catalog = row["prod_catalog"]
        filled_template = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": explanation_template.format(
                            source_text=source_text, prod_catalog=prod_catalog
                        ),
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<explanation>"}],
            },
        ]

        text_explanation = invoke_claude_msg(
            system_prompt, filled_template, model_size, task_type="explanation"
        )
        print(f"| {source_text} | {text_explanation}")

        filled_template = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": translate_template.format(
                            source_text=source_text,
                            prod_catalog=prod_catalog,
                            destination_lang=destination_lang,
                            explanation=text_explanation,
                        ),
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<translation>"}],
            },
        ]
        translate_output = invoke_claude_msg(
            system_prompt, filled_template, model_size, task_type="translation"
        )

        print(f"| {source_text} | {translate_output}")

        new_row = pd.DataFrame(
            {
                "source": [source_text],
                "think": [text_explanation],
                "destination": [translate_output],
            }
        )
        result_df = pd.concat([result_df, new_row], ignore_index=True)

    result_df.to_excel(output_file_path, index=False)


def replace_punctuation(text):
    """
    Replace Chinese commas and periods with their English counterparts.
    """
    text = text.replace("，", ", ").replace("。", "")
    return text


if __name__ == "__main__":
    input_file_path = "./source-text-enriched.xlsx"

    model_size = "sonnet"
    destination_lang = "Japanese"
    output_file_path = (
        "./output/translated-"
        + destination_lang
        + "-"
        + model_size
        + "-"
        + current_time
        + ".xlsx"
    )
    process_excel(input_file_path, output_file_path, destination_lang, model_size)
