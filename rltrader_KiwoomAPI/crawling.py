# import pandas
import pandas as pd

def crawling(code):
    # 해당 종목에 대한 네이버 금융 url 만들기
    url = 'http://finance.naver.com/item/sise_day.nhn?code=' + code

    df = pd.DataFrame()

    # make_url()을 통해 생성된 URL에 Page 번호를 1~30까지 추가하여 크롤링 실시
    for page in range(1, 20):
        # 네이버 금융 url 에 page를 붙여 찾고싶은 page를 추가한다.
        page_url = '{url}&page={page}'.format(url = url, page = page)
        
        # pandas의 df에 page_url을 입력하여 데이터 프레임을 만든다.
        df = df.append(pd.read_html(page_url, header = 0)[0], ignore_index= True)
        
    # Nan 값인 행을 제거한다.
    df = df.dropna()

    ########################################### 출력

    # 한글로 된 열 이름을 영어로 바꾼다.
    df = df.rename(columns= {'날짜': 'date', '종가': 'close', '전일비': 'diff', '시가': 'open', '고가': 'high', '저가': 'low', '거래량': 'volume'})

    # 데이터의 타입을 int형으로 바꾼다.
    df[['close', 'diff', 'open', 'high', 'low', 'volume']]= df[['close', 'diff', 'open', 'high', 'low', 'volume']].astype(int)

    # 열 이름 'date'의 데이터 타입을 date로 바꾼다.
    df['date'] = pd.to_datetime(df['date'])

    df = df.sort_values(by=['date'], ascending=True)

    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

    return df