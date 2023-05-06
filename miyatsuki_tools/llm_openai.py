import json
from json import JSONDecodeError

import openai


def extract_codeblock(output: str):
    if "```" not in output:
        return output

    lines = output.split("\n")

    back_quotes_index = [
        index for index, line in enumerate(lines) if line.startswith("```")
    ]
    start_index = back_quotes_index[-2] + 1
    end_index = back_quotes_index[-1]

    return "\n".join(lines[start_index:end_index])


def correct_json(text: str, model: str):
    prompt = f"""
    以下のJSONをパース可能に修正してください。修正点は記載せず、JSONだけを返してください。
    ```json
    {text}
    ```
    """

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant, which returns answer in JSON format.",
            },
            {"role": "user", "content": prompt[1:-1]},
        ],
        max_tokens=1024,
        temperature=0,
    )

    return response["choices"][0]["message"]["content"]


def parse_llm_output_json(output: str, model: str = "gpt-3.5-turbo"):
    code_block = extract_codeblock(output)

    # まずはそのままパースしてみる
    try:
        return json.loads(code_block)
    except JSONDecodeError:
        pass
    except:
        return None

    # evalでパースしてみる
    try:
        ans = eval(code_block)
        if type(ans) == dict:
            return ans
    except:
        pass

    # それでもダメだったらLLMに直してもらう
    code_block = extract_codeblock(correct_json(code_block, model))
    try:
        return json.loads(code_block)
    except:
        raise ValueError(output)
