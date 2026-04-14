from openai import OpenAI

client = OpenAI(
    api_key="sk-WXtqOuBZPY096KTcDdE866275274464d88943d068aA7Ff5d",
    base_url="https://api.gpt.ge/v1/",
    default_headers={"x-foo": "true"},
)

models = client.models.list()

for m in models.data:
    print(m.id)