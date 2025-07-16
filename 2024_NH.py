#!/usr/bin/env python
# coding: utf-8

# ## **데이터 불러오기**

# In[27]:


import pandas as pd
import numpy as np

# 미리 정제를 위한 함수를 지정하고
import re
def extract_text_only(text):
    # 정규표현식을 사용하여 알파벳이 아닌 문자를 제거
    return ' '.join(re.findall(r'[a-zA-Z]+', text))

# ETF 데이터 불러오기 
df1 = pd.read_csv('DACON_ETF/NH_CONTEST_DATA_ETF_HOLDINGS.csv', encoding='euc-kr')

# 컬럼 이름 변환
df1 = df1.rename(columns={
    'etf_tck_cd': 'ETF',
    'tck_iem_cd': '구성종목',
    'mkt_vlu': '종목가치',
    'fc_sec_eng_nm': '영문명',
    'fc_sec_krl_nm': '한글명',
    'stk_qty': '주수',
    'wht_pct': '비중',
    'sec_tp': '타입'
})

# 데이터 정제
df1['ETF'] = df1['ETF'].apply(extract_text_only)

# ETF 배당 데이터 불러오기
df2 = pd.read_csv('DACON_ETF/NH_CONTEST_DATA_HISTORICAL_DIVIDEND.csv', encoding='euc-kr') 

# 컬럼 이름 변환
df2 = df2.rename(columns={
    'etf_tck_cd': 'ETF',
    'ediv_dt': '배당락일',
    'ddn_amt': '배당금',
    'aed_stkp_ddn_amt': '수정배당금',
    'ddn_bse_dt': '배당 기준일',
    'ddn_pym_dt': '지급일',
    'pba_dt': '공시일',
    'ddn_pym_fcy_cd': '배당주기'
})

# 데이터 정제
df2['ETF'] = df2['ETF'].apply(extract_text_only)

# ETF 점수 데이터 가져오기
df3 = pd.read_csv('DACON_ETF/NH_CONTEST_ETF_SOR_IFO.csv', encoding='euc-kr')

# 컬럼 이름 변환
df3 = df3.rename(columns={
    'bse_dt': '거래일자',
    'etf_iem_cd': 'ETF',
    'mm1_tot_pft_rt': '1개월총수익율',
    'mm3_tot_pft_rt': '3개월총수익율',
    'yr1_tot_pft_rt': '1년총수익율',
    'etf_sor': '점수',
    'etf_z_sor': 'Z점수',
    'z_sor_rnk': 'Z점수순위',
    'acl_pft_rt_z_sor': '누적수익율Z점수',
    'ifo_rt_z_sor': '정보비율Z점수',
    'shpr_z_sor': '샤프지수Z점수',
    'crr_z_sor': '상관관계Z점수',
    'trk_err_z_sor': '트래킹에러Z점수',
    'mxdd_z_sor': '최대낙폭Z점수',
    'vty_z_sor': '변동성Z점수'
})

# 데이터 정제
df3['ETF'] = df3['ETF'].apply(extract_text_only)

# 주식 거래 데이터 가져오기
df4 = pd.read_csv('DACON_ETF/NH_CONTEST_STK_DT_QUT.csv', encoding='euc-kr')

# 컬럼 이름 변환
df4 = df4.rename(columns={
    'bse_dt': '거래일자', 'tck_iem_cd': '종목',
    'iem_ong_pr': '종목시가', 'iem_hi_pr': '종목고가',
    'iem_low_pr': '종목저가', 'iem_end_pr': '종목종가',
    'bf_dd_cmp_ind_pr': '전일대비증감가격', 'bf_dd_cmp_ind_rt': '전일대비증감율',
    'acl_trd_qty': '누적거래수량', 'trd_cst': '거래대금',
    'sll_cns_sum_qty': '매도체결합계수량', 'byn_cns_sum_qty': '매수체결합계수량',
    'sby_bse_xcg_rt': '환율'
})

# 데이터 정제
df4['종목'] = df4['종목'].apply(extract_text_only)

# 주식 정보 데이터 가져오기
df5 = pd.read_csv('DACON_ETF/NH_CONTEST_NHDATA_STK_DD_IFO.csv')

df5 = df5.rename(columns={
    'bse_dt': '일자', 'tck_iem_cd': '종목',
    'tot_hld_act_cnt': '총보유계좌수', 'tot_hld_qty': '총보유수량',
    'tco_avg_hld_qty': '당사평균보유수량', 'tco_avg_hld_wht_rt': '당사평균보유비중비율',
    'tco_avg_eal_pls': '당사평균평가손익', 'tco_avg_phs_uit_pr': '당사평균매입단가',
    'tco_avg_pft_rt': '당사평균수익율', 'tco_avg_hld_te_dd_cnt': '당사평균보유기간일수',
    'dist_hnk_pct10_nmv': '상위10', 'dist_hnk_pct30_nmv': '상위30',
    'dist_hnk_pct50_nmv': '상위50', 'dist_hnk_pct70_nmv': '상위70',
    'dist_hnk_pct90_nmv': '상위90', 'bse_end_pr': '기준종가',
    'lss_ivo_rt': '손실투자자비율', 'pft_ivo_rt': '수익투자자비율',
    'ifw_act_cnt': '신규매수계좌수', 'ofw_act_cnt': '전량매도계좌수',
    'vw_tgt_cnt': '종목조회건수', 'rgs_tgt_cnt': '관심종목등록건수'
})

# 데이터 정제
df5['종목'] = df5['종목'].apply(extract_text_only)

# 고객 데이터 가져오기
df6 = pd.read_csv('DACON_ETF/NH_CONTEST_NHDATA_CUS_TP_IFO.csv')

df6 = df6.rename(columns={
    'bse_dt': '기준일자',
    'tck_iem_cd': '종목',
    'cus_cgr_llf_cd': '대분류',
    'cus_cgr_mlf_cd': '중분류',
    'cus_cgr_act_cnt_rt': '계좌수비율',
    'cus_cgr_ivs_rt': '투자비율'
})

# 데이터 정제
df6['종목'] = df6['종목'].apply(extract_text_only)

# 유입/유출 데이터 가져오기
df7 = pd.read_csv('DACON_ETF/NH_CONTEST_NHDATA_IFW_OFW_IFO.csv')

df7 = df7.rename(columns={
    'bse_dt': '거래일자',
    'tck_iem_cd': '종목',
    'ifw_ofw_dit_cd': '유입/유출구분코드',
    'ifw_ofw_tck_cd': '유입/유출티커코드',
    'ifw_ofw_amt_wht_rt': '유입/유출금액비중',
    'ifw_ofw_rnk': '유입/유출순위'})

df7['종목'] = df7['종목'].apply(extract_text_only)


# ## **단기수익지표 정리**
# - 1개월 수익율 / 3개월 수익율 / MACD / RSI

# In[28]:


# 1. 1개월 수익율과 3개월 수익율의 지수이동평균

# 필요한 컬럼만 가져와서 ETF/거래일자별로 정렬 
단기수익 = df3[['거래일자', 'ETF', '1개월총수익율', '3개월총수익율']].sort_values(['ETF', '거래일자'])
단기수익.set_index('거래일자', inplace=True)

# 1개월 수익율과 3개월 수익율의 지수이동평균 산출
span = 60
단기수익['1개월총수익율_EMA'] = 단기수익['1개월총수익율'].ewm(span=span, adjust=False).mean()
단기수익['3개월총수익율_EMA'] = 단기수익['3개월총수익율'].ewm(span=span, adjust=False).mean()
단기수익_EMA = 단기수익.groupby('ETF').last().reset_index()
단기수익_EMA = 단기수익_EMA[['ETF','1개월총수익율_EMA','3개월총수익율_EMA']]

# 2. 매수 타이밍을 정하깅 위한 MACD, RSI 

# MACD(이동평균선을 통해 추세의 신호를 확인하는 지표) 산출
def MACD(df) :
    df['EMA_12'] = df['종목종가'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['종목종가'].ewm(span=26, adjust=False).mean()

    # MACD와 Signal Line 계산
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df = df.drop(['EMA_12', 'EMA_26'], axis=1)
    return df

# RSI(과매수/과매도 상태를 확인하는 지표) 산출
def RSI(df) :
    delta = df['종목종가'].diff()

    # 상승폭과 하락폭 구하기
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # AU, AD 계산
    window = 14  # RSI 기간
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # RS와 RSI 계산
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# MACD와 RSI 생성을 위한 일자별 종가 생성
etf_list = df1['ETF'].unique().tolist()
일자별종가 = df4[df4['종목'].isin(etf_list)][['거래일자', '종목', '종목종가']].sort_values(['종목', '거래일자'])
일자별종가.set_index('거래일자', inplace=True)
일자별종가

# MACD 컬럼 생성 (가장 마지막 일자의 MACD로)
macd_list = []
for i in etf_list:
    df_종가 = 일자별종가[일자별종가['종목'] == i].copy()  # 복사하여 원본 데이터프레임을 보호
    if not df_종가.empty:  
        df_macd = MACD(df_종가)
        macd_list.append(df_macd['MACD'].iloc[-1])  # 마지막 MACD 값으로 
단기['MACD'] = macd_list

# RSI 생성 (가장 마지막 일자의 RSI로)
RSI_list = []
for i in etf_list:
    df_종가 = 일자별종가[일자별종가['종목'] == i].copy()  # 복사하여 원본 데이터프레임을 보호
    if not df_종가.empty:  
        df_rsi = RSI(df_종가)
        RSI_list.append(df_rsi['RSI'].iloc[-1]) # 마지막 RSI 값으로 
        
단기['RSI'] = RSI_list      

# 데이터 병합 및 단기지표 데이터 완성
단기수익지표 = 단기수익_EMA.merge(단기, on='ETF', how='left')
단기수익지표


# ## **장기수익지표 정리**
# - 1년 수익률 / 누적 수익률 / Z점수 / 정보 비율 / 샤프지수 / 1년_배당 수익

# In[30]:


# 1. 1년 수익률과 누적 수익율의 지수이동평균 

# 1년 수익률과 누적 수익룰 점수 불러오기
장기수익 = df3[['거래일자', 'ETF', '1년총수익율', '누적수익율Z점수']].sort_values(['ETF', '거래일자'])
장기수익.set_index('거래일자', inplace=True)

# 1년 수익율과 누적 수이율의 지수이동평균 계산
span = 60
장기수익['1년수익율_EMA'] = 장기수익['1년총수익율'].ewm(span=span, adjust=False).mean()
장기수익['누적수익율Z점수_EMA'] = 장기수익['누적수익율Z점수'].ewm(span=span, adjust=False).mean()
장기수익_EMA = 장기수익.groupby('ETF').last().reset_index()
장기수익지표 = 장기수익_EMA[['ETF', '1년수익율_EMA','누적수익율Z점수_EMA']]

# 2. 정보비율과 샤프지수

# 정보비율과 샤프지수 데이터 불러오기
정보_샤프 = df3[['거래일자', 'ETF', '정보비율Z점수', '샤프지수Z점수']].sort_values(['ETF', '거래일자'])
정보_샤프.set_index('거래일자', inplace=True)

# 정보비율과 샤프지수의 지수이동평균 계산
span = 60
정보_샤프['정보비율_EMA'] = 정보_샤프['정보비율Z점수'].ewm(span=span, adjust=False).mean()
정보_샤프['샤프지수_EMA'] = 정보_샤프['샤프지수Z점수'].ewm(span=span, adjust=False).mean()
정보_샤프_EMA = 정보_샤프.groupby('ETF').last().reset_index()
정보_샤프_EMA = 정보_샤프_EMA[['ETF', '정보비율_EMA', '샤프지수_EMA']]

# 데이터 1차 병합
장기수익지표 = 장기수익지표.merge(정보_샤프_EMA, on='ETF')

# 3. 1년 배당수익

# 배당 데이터 불러오기
배당 =  df2[['ETF', '배당락일', '수정배당금', '배당주기']].sort_values(['ETF', '배당락일']).groupby('ETF').last().reset_index()
배당['배당주기'] = 배당['배당주기'].map({'Annual' : 1, 'Quarterly' : 4, 'Monthly' : 12})

# 1년 기준 배당 수익 계산
배당['1년_배당수익'] = 배당['수정배당금'] * 배당['배당주기'] 

# 데이터 2차 병합
장기수익지표 =  장기수익지표.merge(배당[['ETF', '1년_배당수익']], on ='ETF', how='inner')
장기수익지표


# ## **변동성 지표 정리**
# - 변동성 / 최대낙폭 / 상관관계 / 트래킹에러 / 볼린저갭 / 거래량 변동성

# In[33]:


# 1. 변동성 / 최대낙폭 / 상관관계 / 트래킹에러의 지수이동평균

# 혼합지표 중 변동성 데이터 불러오기
변동성 = df3[['거래일자', 'ETF', '변동성Z점수', '최대낙폭Z점수', '상관관계Z점수', '트래킹에러Z점수']].sort_values(['ETF', '거래일자'])
변동성.set_index('거래일자', inplace=True)

# 변동성 / 최대낙폭 / 상관관계 / 트래킹에러의 지수이동평균 산출
span = 60
변동성['변동성Z점수_EMA'] = 변동성['변동성Z점수'].ewm(span=span, adjust=False).mean()
변동성['최대낙폭Z점수_EMA'] = 변동성['최대낙폭Z점수'].ewm(span=span, adjust=False).mean()
변동성['상관관계Z점수_EMA'] = 변동성['상관관계Z점수'].ewm(span=span, adjust=False).mean()
변동성['트래킹에러Z점수_EMA'] = 변동성['트래킹에러Z점수'].ewm(span=span, adjust=False).mean()
변동성_z = 변동성.groupby('ETF').last().reset_index()
변동성_z = 변동성_z[['ETF','변동성Z점수_EMA', '최대낙폭Z점수_EMA', '상관관계Z점수_EMA', '트래킹에러Z점수_EMA']]

# 2. 볼린저밴드의 상한과 하한 간의 거리 (볼린저갭)

# 볼린저 밴드 함수 정의
def Bollinger(df) : 
    df['Middle_Band'] = df['종목종가'].rolling(window=20).mean()

    # 표준편차 계산
    df['Standard_Deviation'] = df['종목종가'].rolling(window=20).std()

    # 상단 밴드와 하단 밴드 계산
    df['Upper_Band'] = df['Middle_Band'] + (df['Standard_Deviation'] * 2)
    df['Lower_Band'] = df['Middle_Band'] - (df['Standard_Deviation'] * 2)
    df['gap'] = df['Upper_Band'] - df['Lower_Band']
    return df

# 볼린저 밴드 함수를 이용해 볼린저 갭 산출
gap_list = []
for i in etf_list:
    df_종가 = 일자별종가[일자별종가['종목'] == i].copy()  # 복사하여 원본 데이터프레임을 보호
    if not df_종가.empty:  #
        df_gap = Bollinger(df_종가)
        gap_list.append(df_gap['gap'].iloc[-1]) # 가장 마지막 날짜의 볼린저갭으로
볼린저갭 = pd.DataFrame({'ETF' : etf_list, '볼린저갭' : gap_list}) 

# 데이터 1차 병합
변동지표 = 변동성_z.merge(볼린저갭, on = 'ETF', how='left')

# 거래량_변동성 만들기
거래_변동 = df4[['거래일자', '종목','종목종가', '누적거래수량']].sort_values(['종목', '거래일자'])
거래_변동.set_index('거래일자', inplace=True)

# 거래변동성 산출
거래_변동 = 거래_변동.groupby('종목')['누적거래수량'].std().reset_index(name='거래_변동성')

변동지표 = 변동지표.merge(거래_변동, left_on = 'ETF', right_on='종목')
변동지표 = 변동지표.drop('종목', axis=1)

변동지표


# ## **유동성 지표 만들기**
# - 총거래금액 / 매도매수_차이 / 유입변동성 / 유출변동성

# In[35]:


# 유입/유출 데이터 필요한 부분만 불러오기
유동 = df4[['거래일자', '종목','종목종가', '누적거래수량', '거래대금', '매도체결합계수량', '매수체결합계수량']].sort_values(['종목', '거래일자'])
유동.set_index('거래일자', inplace=True)

# 1. 거래 규모
# 60일 간의 거래대금을 모두 합하여 60일 간의 거래규모 산출
유동_거래_규모 = 유동.groupby('종목')['거래대금'].sum().reset_index(name='총거래금액')

# 2. 매수-매도 비율 차이
# 매수 수량과 매도 수량의 차이를 누적거래수량으로 나눴을 떄의 차이의 지수이동평균 _ 0에 가까울수록 균형
유동['매수-매도_비율'] =  (유동['매수체결합계수량'] - 유동['매도체결합계수량']) / 유동['누적거래수량']
span = 60
유동['매수매도_차이비율_EMA'] = 유동['매수-매도_비율'].ewm(span=span, adjust=False).mean()

# ETF별로 마지막 일자의 행만 가져와 1차 결합
유동 = 유동.groupby('종목').last().reset_index()[['종목', '매수매도_차이비율_EMA']]
유동지표 = 유동_거래_규모.merge(유동, on='종목')

# 3. 유입/유출의 변동성

# 유입/유출 데이터 정렬
유동_분산 = df7.sort_values(['종목', '거래일자'])
유동_분산.set_index('거래일자', inplace=True)

# 매수금액/매도금액 계산
df4['매수총금액'] = df4['종목종가'] * df4['매수체결합계수량']
df4['매도총금액'] = df4['종목종가'] * df4['매도체결합계수량']

# 유입/유출 데이터와 매수/매도 금액 데이터 결합
유동_분산 = df7.merge(df4[['거래일자', '종목', '매수총금액', '매도총금액']], on = ['거래일자', '종목'])
유동_분산 = 유동_분산.merge(df1[['ETF', '종목가치']], left_on='종목', right_on='ETF').drop('ETF', axis=1)

# 일자별 비중을 통해 ETF/일자별로 유입/유출 종목과 금액계산
유동_분산['유입/유출금액'] = 유동_분산.apply(lambda row: row['유입/유출금액비중'] * row['매수총금액'] if row['유입/유출구분코드'] == 1 
                                 else row['유입/유출금액비중'] * row['매도총금액'], axis=1)
유동_분산['유입/유출금액'] = 유동_분산['유입/유출금액'] / 100
유동_분산 = 유동_분산.drop(['매수총금액', '매도총금액'],axis=1).set_index(['종목', '거래일자'])

# 일자별로 유입/유출금액의 편차 산출
유동_분산['유입/유출_변동성'] = 유동_분산.groupby(['거래일자', '종목', '유입/유출구분코드'])['유입/유출금액'].transform('std').fillna(0)
유동_분산['유입/유출_변동성'] = 유동_분산['유입/유출_변동성'] / 유동_분산['종목가치']

# 유입/유출 데이터 구분
유동_분산_유입 = 유동_분산[유동_분산['유입/유출구분코드']==1]
유동_분산_유출 = 유동_분산[유동_분산['유입/유출구분코드']==2]

# 유입금액과 유출금액의 편차의 변동성 계산
span = 60
유동_분산_유입['유입변동성_EMA'] = 유동_분산_유입['유입/유출_변동성'].ewm(span=span, adjust=False).mean()
유동_분산_유출['유출변동성_EMA'] = 유동_분산_유출['유입/유출_변동성'].ewm(span=span, adjust=False).mean()

# ETF별로 마지막 일자의 행만 가져와 2차 결합
유동_분산_유입 = 유동_분산_유입.groupby('종목').last().reset_index()[['종목', '유입변동성_EMA']]
유동_분산_유출 = 유동_분산_유출.groupby('종목').last().reset_index()[['종목', '유출변동성_EMA']]
유동_분산 = 유동_분산_유입.merge(유동_분산_유출, on = '종목')

# 최종데이터 생성
유동지표 = 유동지표.merge(유동_분산, on='종목', how='left')
유동지표 = 유동지표.rename(columns={'종목' : 'ETF'})
유동지표


# ## **관심도 지표 만들기**
# - 종목조회건수 / 관심종목등록건수 / 계좌수 증감률 / 관심등록대비 신규매수계좌

# In[36]:


# 필요한 데이터 불러오기
관심도 = df5[['일자', '종목', '총보유계좌수', '상위10', '상위90', '기준종가', '수익투자자비율', '신규매수계좌수', '전량매도계좌수', '종목조회건수', '관심종목등록건수']].sort_values(['종목', '일자'])
관심도.set_index('일자', inplace=True)

# 1. 60일 간의 종목조회수, 관심종목등록건수 합계
# 종목조회수, 관심등록수, 신규매수계좌수의 합계 구하기
관심도_sum = 관심도.groupby('종목')[['종목조회건수', '관심종목등록건수', '신규매수계좌수']].sum()

# 2. 계좌수 증감률의 지수이동평균
# 60일 간의 계좌수 증감률을 구한뒤 지수이동편균 산출
관심도['계좌수_증감률'] = 관심도['총보유계좌수'].pct_change() * 100
관심도['계좌수_증감률_EMA'] = 관심도['계좌수_증감률'].ewm(span=60, adjust=False).mean()
관심도_updown = 관심도.groupby('종목').last().reset_index()
관심도_updown = 관심도_updown[['종목', '계좌수_증감률_EMA']]

# 데이터 1차 결합
관심지표 = 관심도_sum.merge(관심도_updown, on='종목')

# 3. 관심등록대비_신규매수계좌수 
# 앞서 구한 관심종목등록수 합계와 신규매수계좌수 합계 활용
관심지표['관심대비매수'] = 관심지표['신규매수계좌수'] / 관심지표['관심종목등록건수']
관심지표 = 관심지표.rename(columns={'종목' : 'ETF'})
관심지표 = 관심지표.drop('신규매수계좌수', axis=1)
관심지표


# ## **데이터병합**

# In[37]:


# 데이터 병합
df_final = 단기수익지표.merge(장기수익지표, on='ETF')
df_final = df_final.merge(변동지표, on='ETF', how='left')
df_final = df_final.merge(유동지표, on='ETF')
df_final = df_final.merge(관심지표, on='ETF')

df_final.columns = ['ETF', '1개월_수익율', '3개월_수익율', 'MACD', 'RSI',
       '1년_수익률', '누적수익율(Z)', '정보비율(Z)', '샤프지수(Z)', '1년_배당수익', '변동성(Z)', '최대낙폭(Z)', '상관관계(Z)','트래킹에러(Z)',
       '볼린저갭', '거래_변동성', '총거래금액', '매수매도_차이비율', '유입금액_분산', '유출금액_분산', '종목조회수', '관심종목등록수', '계좌수증감률', '관심등록대비_매수계좌']
df_final


# In[38]:


# 결측치 계산
df_final.isna().sum()


# In[157]:


# 결측치 처리 : MICE
from fancyimpute import IterativeImputer
import pandas as pd

# 같은 카테고리의 변수로 MICE 결측치 대체
columns_to_use1 = ['1개월_수익율', '3개월_수익율', 'MACD', 'RSI']
columns_to_use2 = ['변동성(Z)', '최대낙폭(Z)', '볼린저갭', '거래_변동성', '상관관계(Z)', '트래킹에러(Z)']
columns_to_use3 = ['총거래금액', '매수매도_차이비율', '계좌수증감률', '유입금액_분산', '유출금액_분산']

df_subset1 = df_final[columns_to_use1]
df_subset2 = df_final[columns_to_use2]
df_subset3 = df_final[columns_to_use3]

# MICE 방식으로 결측치 대체
imputer = IterativeImputer()
df_imputed1 = imputer.fit_transform(df_subset1)
df_imputed2 = imputer.fit_transform(df_subset2)
df_imputed3 = imputer.fit_transform(df_subset3)

# 대체한 값을 원래 데이터프레임에 다시 넣음
df_final[['1개월_수익율', '3개월_수익율', 'MACD', 'RSI']] = pd.DataFrame(df_imputed1, columns=columns_to_use1)
df_final[['변동성(Z)', '최대낙폭(Z)', '볼린저갭', '거래_변동성', '상관관계(Z)', '트래킹에러(Z)']] = pd.DataFrame(df_imputed2, columns=columns_to_use2)
df_final[['총거래금액', '매수매도_차이비율', '계좌수증감률', '유입금액_분산', '유출금액_분산']] = pd.DataFrame(df_imputed3, columns=columns_to_use3)

# 결과 확인
df_final


# In[158]:


df_final.to_csv('ETF데이터프레임_1차.csv', index=False)


# In[ ]:




