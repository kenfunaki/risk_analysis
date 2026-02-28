import pandas as pd
import random
from datetime import datetime
import random
from datetime import datetime, timedelta
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


#UI画面でExcelを読み込み。
#main.pyの関数内でコール＝＝＞app.pyで実行
def load_master_file(file):
    return pd.read_excel(file)

# 中核的な関数
# 貸借の勘定科目をランダムセレクト、日付を範囲内でランダム作成、摘要欄は後述の関数をコール
# main.pyの関数内でコール＝＝＞app.pyで実行
# 現状ではuser_hintは無駄になっている。
def generate_normal_entry(account_master, user_hint="", start_date=None, end_date=None):
    debit = random.choice(account_master["借方科目"].dropna().tolist())
    credit = random.choice(account_master["貸方科目"].dropna().tolist())
    amount = random.randint(1000, 100000)

        # 日付を期間内でランダム生成
    if start_date and end_date:
        days_range = (end_date - start_date).days
        random_days = random.randint(0, days_range)
        date = (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")
    else:
        date = datetime.now().strftime("%Y-%m-%d")

    description = generate_description_from_account(debit, credit, user_hint)

    return {
        "日付": date,
        "借方科目": debit,
        "貸方科目": credit,
        "金額": amount,
        "摘要": description
    }


#先述の関数でコール
def generate_description_from_account(debit, credit, user_hint=""):
    if user_hint:
        instruction_text = f"次の指示にも従ってください：{user_hint}"
    else:
        instruction_text = ""

    prompt = f"""
以下の仕訳について、架空のイベントや出来事を想定し、文章ではなく簡潔な実務的な摘要を1行で作成してください。
{instruction_text}

借方：{debit}
貸方：{credit}
摘要：
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=50,
    )

    return response.choices[0].message["content"].strip()
