import json
import numpy as np
import re
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import math

# --- 토크나이저 불러오기 ---
def load_tokenizer(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return tokenizer_from_json(json.load(f))

tokenizer_q = load_tokenizer('tokenizer_q.json')
tokenizer_a = load_tokenizer('tokenizer_a.json')

# --- 모델 불러오기 ---
model = load_model('model.h5', compile=False)
# --- 파라미터 설정 ---
max_len_q = model.input_shape[0][1]
max_len_a = model.input_shape[1][1]
index_to_word = {v: k for k, v in tokenizer_a.word_index.items()}
index_to_word[0] = ''  # 패딩 토큰

start_token = 'start'
end_token = 'end'

# --- 생성 모델 응답 (Greedy Search) ---
def decode_sequence_greedy(input_text):
    input_seq = tokenizer_q.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_len_q, padding='post')

    target_seq = tokenizer_a.texts_to_sequences([start_token])[0]
    target_seq = pad_sequences([target_seq], maxlen=max_len_a, padding='post')

    decoded_sentence = ''
    for i in range(max_len_a):
        predictions = model.predict([input_seq, target_seq], verbose=0)
        prob_dist = predictions[0, i, :]
        pred_id = np.argmax(prob_dist)  # 🔥 핵심: 가장 확률 높은 토큰 선택 (Greedy!)

        pred_word = index_to_word.get(pred_id, '')
        if pred_word == end_token:
            break

        decoded_sentence += pred_word + ' '
        if i + 1 < max_len_a:
            target_seq[0, i + 1] = pred_id

    return re.sub(r'\s+', ' ', decoded_sentence).strip()

def extract_main_query(text):
    # 문장 분리 (".", "?" 기준)
    sentences = re.split(r'[.?!]\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return text

    # 마지막 문장만 선택해서 조사 제거
    last = sentences[-1]

    # 괄호/특수문자 제거
    last = re.sub(r'[^가-힣a-zA-Z0-9 ]', '', last)

    # 조사 제거
    particles = ['이', '가', '은', '는', '을', '를', '의', '에서', '에게', '한테', '보다']
    for p in particles:
        last = re.sub(rf'\b(\w+){p}\b', r'\1', last)

    return last.strip()


# --- 위키백과 요약 크롤링 ---
def get_wikipedia_summary(query):
    cleaned_query = extract_main_query(query)
    url = f"https://ko.wikipedia.org/api/rest_v1/page/summary/{cleaned_query}"
    res = requests.get(url)
    if res.status_code == 200:
        return res.json().get("extract", "요약 정보를 찾을 수 없습니다.")
    else:
        return "위키백과에서 정보를 가져올 수 없습니다."

# --- intent 분류기 (룰 기반) ---
def simple_intent_classifier(text):
    text = text.lower()
    greet_keywords = ["안녕", "반가워", "이름", "누구", "소개", "어디서 왔", "정체", "몇 살", "너 뭐야"]
    info_keywords = ["설명", "정보", "무엇", "뭐야", "어디", "누구", "왜", "어떻게", "종류", "개념"]
    math_keywords = ["더하기", "빼기", "곱하기", "나누기", "루트", "제곱", "+", "-", "*", "/", "=", "^", "√", "계산", "몇이야", "얼마야"]

    if any(kw in text for kw in greet_keywords):
        return "인사"
    elif any(kw in text for kw in info_keywords):
        return "정보질문"
    elif any(kw in text for kw in math_keywords):
        return "수학질문"
    else:
        return "일상대화"


def parse_math_question(text):
    # 자연어 → 수식으로 간단 변환
    text = text.replace("곱하기", "*")
    text = text.replace("더하기", "+")
    text = text.replace("빼기", "-")
    text = text.replace("나누기", "/")
    text = text.replace("제곱", "**2")
    text = re.sub(r'루트\s*(\d+)', r'math.sqrt(\1)', text)  # 루트 9 → math.sqrt(9)

    # 숫자만 있는지 확인 후 계산
    try:
        result = eval(text)
        return f"정답은 {result}입니다."
    except:
        return "계산할 수 없는 수식이에요. 다시 한번 확인해 주세요!"


def is_math_question(text):
    return bool(re.search(r'[0-9]+|곱하기|더하기|나누기|빼기|루트|제곱', text))

def respond(input_text):
    intent = simple_intent_classifier(input_text)

    if "이름" in input_text:
        return "제 이름은 TriFusionNet입니다."
    
    if "/사용법" in input_text:
        return "\n 자유롭게 사용해주세요. \n 딱히 제약은 없습니다."
        

    if "/help" in input_text:
        return "1. \n /model : 모델 소개\n /detail : 모델 아키텍쳐 \n /사용법"

    if "/model" in input_text:
       return (
        "저는 TriFusionNet이라는 2개의 디코더와 멀티-헤드 어텐션을 사용하는 Seq2Seq 챗봇입니다.\n"
        "모델 아키텍처 요약:\n"
        "- 입력 레이어: 2개의 입력을 받아 각기 다른 임베딩 레이어로 변환\n"
        "- 인코더: LSTM을 통해 입력을 처리하고, 각기 다른 Dense 레이어로 변환\n"
        "- 디코더: 2개의 LSTM을 병렬로 처리한 후 결합\n"
        "- 멀티-헤드 어텐션: 인코더 출력과 디코더 출력을 결합하여 어텐션을 수행\n"
        "- 출력: TimeDistributed 레이어를 통해 최종 예측값을 출력"
        "\n이 모델은 자연어 처리에서 복잡한 관계를 잘 학습할 수 있는 구조입니다.\n 더 자세한 정보를 원하신다면 /detail을 입력해주세요!"
    )

    if "누구" in input_text:
        return "저는 TriFusionNet이라고 해요. "

    if intent == "수학질문":
        return parse_math_question(input_text)

    if intent == "정보질문":
        keyword = re.sub(r"(에 대해|에 대한|에 대해서)?\s*(설명해줘|알려줘|뭐야|개념|정의|정보)?", "", input_text).strip()
        if not keyword:
            return "어떤 주제에 대해 궁금한가요?"
        summary = get_wikipedia_summary(keyword)
        return f"{summary}\n다른 궁금한 점 있으신가요?"
    else:
        return decode_sequence_greedy(input_text)

def auto_linebreak(text, max_length=80):
    """
    마침표, 느낌표, 물음표 뒤에 줄바꿈을 자동으로 추가하는 함수.
    쉼표는 줄바꿈하지 않음.
    """
    # 1. 마침표, 느낌표, 물음표 뒤에 줄바꿈
    text = re.sub(r'(?<=[.!?])\s+', '\n', text)

    # 2. 길이가 너무 긴 줄은 적당히 자르기 (선택)
    if max_length:
        lines = []
        for line in text.split('\n'):
            while len(line) > max_length:
                cut = line[:max_length].rfind(' ')
                if cut == -1:
                    cut = max_length
                lines.append(line[:cut])
                line = line[cut:].lstrip()
            lines.append(line)
        text = '\n'.join(lines)

    return text


# --- 대화 루프 ---
if __name__ == "__main__":
    print("🤖 TriFusionNet 챗봇입니다! 궁금한 점이 있다면 /help를 입력해보세요 (종료: 'exit')\n")

    while True:
        input_text = input("👤 당신: ")
        if input_text.lower() == 'exit':
            print("👋 안녕히 가세요!")
            break

        output_text = respond(input_text)
        formatted_output = auto_linebreak(output_text)  # ← 여기서 줄바꿈 적용!
        print("🤖 TriFusionNet:\n" + formatted_output)  # ← 보기 좋게 개행해서 출력

