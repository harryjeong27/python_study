# -------------------------------- ADVANCED --------------------------------- #

# =============================================================================
# R, Python 적용함수 비교
#                          R             Python
# 1차원 원소별 적용       sapply         map(), .map() 
# 2차원 행,열별 적용      apply            .apply
# 2차원 원소별 적용       apply           .applymap
# =============================================================================

# 고급 1. python에서의 적용 함수 / 메서드
# 1) map 함수
# - 1차원에 원소별 적용 가능
# - 반드시 리스트로만 출력 가능
# - 적용함수의 input은 스칼라 형태
# - 함수의 추가 인자 전달 가능

# 2) map 메서드 (pandas 제공 메서드)
# - 1차원(시리즈)에 원소별 적용 가능
# - 반드시 시리즈로만 출력 가능
# - 적용함수의 input은 스칼라 형태
# - 함수의 추가 인자 전달 불가능

# 두 map 함수의 차이
s1 = Series(['abc', 'bcd'])
map(arg, *iterables)             # 함수
s1.map(arg, na_action = None)    # 메서드 

# 3) apply (apply in R)
# - 2차원 데이터프레임의 행별, 컬럼별 적용 가능 (pandas 제공 메서드)
# - 적용함수의 input은 그룹(여러 개 값을 갖는) 형태
# - 함수의 추가 인자 전달 가능 (함수의 옵션은 전달 가능) => 추가 키워드 인자로 전달 가능 (n = 3)

# 4) applymap (sapply in R)
# - 2차원 데이터프레임의 원소별 적용 가능 (pandas 제공 메서드)
# - 출력 결과 데이터프레임
# - 적용함수의 input은 스칼라 형태
# - 함수의 추가 인자 전달 불가
pd.read_csv(filename,
            sep = ',',
            skiprows = 1,
            encoding = 'euc-kr',
            engine = 'python',
            na_values = ['?', '.', '-'])    # 리스트 내 문자들이 na로 처리

df1 = pd.read_csv('emp.csv')
df1.dtypes

df1.EMPNO

# 'nan', 'null' 문자열은 자동으로 nan 처리 => float로 받아줌
# --------------------------------------------------------------------------- #

# 고급 2. Multi-Index
# - index가 여러 층(level)을 갖는 형태
# - 파이썬 Multi-Index 지원 (R에서는 불가)
# - 각 층은 level로 선택 가능 (상위레벨(0))

# 2.1 Multi-Index 생성
#      col1    col2
# A a   
#   b   3
# B a
#   b
# 맨 왼쪽이 상위레벨, 그다음이 하위레벨

df11 = DataFrame(np.arange(1, 9).reshape(4, 2))

df11.index = ['a', 'b', 'c', 'd']
# df11.index = [[상위레벨값], [하위레벨값]]
df11.index = [['A', 'A', 'B', 'B'], ['a', 'b', 'a', 'b']]

df11.index    # MultiIndex
df11.columns = ['col1', 'col2']
df11.index.names = ['상위레벨', '하위레벨']

# 예제) 다음의 데이터 프레임 생성
#      col_a           col_b
#      col1    col2    col1    col2
# A a  1        2       3       4
#   b  5        6       7       8
# B a  9        10      11      12
#   b  13       14      15      16
df12 = DataFrame(np.arange(1, 17).reshape(4, 4))
df12.index = [['A', 'A', 'B', 'B'], ['a', 'b', 'a', 'b']]
df12.columns = [['col_a', 'col_a', 'col_b', 'col_b'], ['col1', 'col2', 'col1', 'col2']]

df22 = DataFrame(np.arange(1,17).reshape(4,4),
                 index = [['A','A','B','B'], ['a','b','a','b']],
                 columns = [['col_a','col_a','col_b','col_b'],
                            ['col1','col2','col1','col2']])

# 2.2 색인
# 2.2.1 iloc
# - 특정 위치 값 선택 가능
df22['col_a']   # 상위 컬럼의 key 색인 가능
df22['col1']    # 하위 컬럼 에러발생
df22.iloc[:, 0]  # 멀티 인덱스의 위치값 색인 가능

# 2.2.2 loc
# - 상위레벨의 색인 가능
# - 하위레벨의 값 선택 불가
# - 인덱스 값의 순차적 전달을 통한 색인 가능
df22.loc[:,'col_a'] # 상위 컬럼의 이름 색인 가능
df22.loc[:,'col1']  # 하위 컬럼의 이름 색인 불가
df22.loc['A',:]     # 상위 인덱스의 이름 색인 가능
df22.loc['a',:]     # 하위 인덱스의 이름 색인 불가

# 예) df3에서 'A' 선택
df2 = DataFrame(np.arange(1,9).reshape(2,4),
                columns=[['col1','col1','col2','col2'],
                         ['A','B','A','B']])
df3 = df2.stack()

df3.loc[[(0,'A'),(1,'A')],:]

# 2.2.3 멀티 인덱스 색인 메서드 : xs(함수)
# - 하위레벨 색인 가능
# - 인덱스의 순차적 전달 없이도 바로 하위 레벨 색인 가능
# - 특정 레벨을 선택한 전달 가능(중간 레벨 스킵 가능)
df22.iloc[]
df22.loc[]
df22.xs('col2', axis=1, level=1).xs('b', axis=0, level=1)

df22.loc[:, ('col_a','col1')]          # 상위부터 하위까지의 순차적 색인은 loc 가능
df22.loc[('A','a'), ('col_a','col1')]  # 상위부터 하위까지의 순차적 색인은 loc 가능

# 예) df3에서 'A' 선택
df3.xs('A', axis=0, level=1)

# 예) 3개 레벨을 갖는 아래 df33에서 index가 'A' 이면서 '1'인 행 선택
df33 = DataFrame(np.arange(1,17).reshape(8,2),
                 index=[['A','A','A','A','B','B','B','B'],
                        ['a','a','b','b','a','a','b','b'],
                        ['1','2','1','2','1','2','1','2']])

df33.loc[('A','1'),:]    # 중간 레벨 생략 불가
df33.loc[('A',:,'1'),:]  # 중간 레벨 생략 불가
df33.loc[[('A','a','1'),('A','b','1')],:]  # 중간 레벨 생략 불가

df33.xs(('A','1'), level=[0,2])    # 중간 레벨 생략 가능

# 2.2.4 get_level_values
# - index object의 method
# - 선택하고자 하는 레벨 이름 혹은 위치값 전달
# - index의 특정 레벨 선택 후 조건 전달 방식

# 예) df33에서 세번째 레벨의 값이 '1'인 행 선택
df33.xs('1', axis=0, level=2)

df33.loc[df33.index.get_level_values(2) == '1', :]

# 2.3 산술 연산
# 1) multi-index의 axis만 전달 시 : multi-index 여부와 상관없이 
#    axis=0 : 행별(세로방향)
#    axis=1 : 컬럼별(가로방향)
df1.sum(axis=0)
df1.sum(axis=1)

# 2) multi-index의 axis, level 동시 전달 시
#    multi-index의 각 레벨이 같은 값끼리 묶여 그룹 연산 

# 예) 지역별 판매량 총합
df1.sum(axis=1, level=0)

# 예) 구분별(컴퓨터,가전,모바일) 판매량 총합
df1.sum(axis=0, level=0)

# 2.4 정렬
# 1) index 순 정렬
df1.sort_index(axis=0, level=0)  # 구분 순서
df1.sort_index(axis=1, level=0)  # 지역 순서
df1.sort_index(axis=1, level=1, ascending=False)  # 지역 순서 역순
df1.sort_index(axis=1, level=[0,1], ascending=[True, False])  # 지역 순서 역순

# 2) 특정 컬럼 값순 정렬 : 컬럼의 이름을 튜플로 전달
df1.sort_values(by=('서울','A'), ascending=False)
df1.sort_values(by=[('서울','A'),('경기','B')], ascending=False)

# 2.5 level 치환
card.T
card.swapaxes(1,0)

df1.swaplevel(1,0, axis=1)
# --------------------------------------------------------------------------- #

# 고급 3.
# 3.1 stack과 unstack
# - stack : wide -> long(tidy data)
# - unstack : long -> wide(cross table)

# multi-index의 stack과 unstack

# R에서의 stack과 unstack : 컬럼 단위
# python에서의 stack과 unstack : index(column 포함) 단위

# 3.1.1 Series에서의 unstack : index의 값을 컬럼화(DataFrame 리턴)
s1 = Series([1,2,3,4], index=[['A','A','B','B'],['a','b','a','b']])

s1.unstack()        # index의 가장 하위 level이 unstack 처리
s1.unstack(level=0) # 지정한 index의 level이 unstack 처리

# 3.1.2 DataFrame에서의 stack : 싱글컬럼의 값을 index화(Series 리턴)
#                              멀티컬럼의 값을 index화(DataFrame 리턴)

df1 = s1.unstack() 
df1.stack()         # 하위 컬럼의 값이 stack 처리
df1.stack(level=0)  # 지정한 column의 level이 stack 처리

df2 = DataFrame(np.arange(1,9).reshape(2,4),
                columns=[['col1','col1','col2','col2'],
                         ['A','B','A','B']])

df2.stack()
df2.stack(level=0)

# 3.1.3 DataFrame에서의 unstack : 특정 레벨의 index를 컬럼화
df3 = df2.stack()
df3.unstack()
df3.unstack(level=0)

# 3.2 cross-table
# - wide data
# - 행별, 열별 정리된 표 형식 => 행별, 열별 연산 용이
# - join 불가
# - group by 연산 불가
# - 시각화시 주로 사용
# - multi-index를 갖는 구조를 unstack 처리하여 얻거나 pivot 통해 가능

# 예)     
# 부서     A  B  C    
# 성별   
# 남      90 89 91
# 여      89 78 95

# 3.2.1 pivot
# - 각 컬럼의 값을 교차테이블 구성요소로 전달, 교차테이블 완성
# - index, columns, values 컬럼 각각 전달
# - grouping 기능 없음(agg func)
# - index, columns 리스트 전달 불가
# - values 리스트 전달 가능

# 3.2.2 pivot_table
# - 교차 테이블 생성 메서드
# - values, index, columns 컬럼 각각 전달 (순서유의)
# - 결합기능(aggregate function 가능 (default : mean))
# - values, index, columns컬럼에 리스트 전달 가능

# [ 예제 : 아래 데이터 프레임을 각각 교차 테이블 형태로 정리 ]
pv1 = pd.read_csv('dcast_ex1.csv', engine='python')
pv2 = pd.read_csv('dcast_ex2.csv', engine='python')
pv3 = pd.read_csv('dcast_ex3.csv', engine='python', encoding = 'euc-kr')

# 1) pv1에서 품목별 price, qty 정보를 정리한 교차표
pv1.pivot(index='name', columns='info', values='value')
pv1.set_index(['name','info'])['value'].unstack()

# 2) pv2에서 년도별, 품목별 판매현황 정리
pv2.pivot('year','name',['qty', 'price'])

# 3) pv3에서 년도별 음료의 판매현황(수량) 정리
pv3.pivot('년도','이름','수량')           # 중복 값이 있어 불가
pv3.pivot(['년도','지점'],'이름','수량')   # 리스트 전달 불가

pv3.pivot_table(index=['년도','지점'],   # 리스트 전달 가능
                columns='이름',
                values='수량')

pv3.pivot_table('수량',['년도','지점'],'이름')  # 인자 이름 생략 시
                                            # values, index, columns순

pv3.pivot_table('수량','년도','이름')                 # 요약기능 가능
pv3.pivot_table('수량','년도','이름', aggfunc='sum')  # 요약함수 전달 가능

# 3.3 rank 메서드
# - 순위 출력 함수
# - R과 비슷
# - pandas 제공
# - axis 옵션 가능 : 자체 행별, 열별 적용 가능
s1 = Series([10,2,5,1,6])
s2 = Series([10,2,5,1,1,6])

s1.rank(axis,              # 진행 방향
        method={'average', # 서로 같은 순위 부여, 평균값으로
                'min',     # 서로 같은 순위 부여, 순위중 최소값으로
                'max',     # 서로 같은 순위 부여, 순위중 최대값으로
                'first'},  # 서로 다른 순위 부여, 앞에 있는 관측치에 더 높은순위
        ascending)         # 정렬 순서
s1
s1.rank()
s2
s2.rank(method='first')

# 1) 순위        1 2 3 4 5 6
s3 = Series([1,2,2,2,3,4])
s3.rank()
s3.rank(method='min')

# 2) DataFrame의 rank 사용
df1 = DataFrame({'col1':[4,1,3,5], 'col2':[1,2,3,4]})

df1.rank(axis=0)  # 세로방향, 같은 컬럼 내 순위 부여
df1.rank(axis=1)  # 가로방향, 같은 행 내 순위 부여

# 3.4 merge 
# - 두 데이터의 join
# - 세개 이상의 데이터의 join 불가
# - equi join만 가능
# - outer join 가능
pd.merge(left,              # 첫번째 데이터 셋
         right,             # 두번째 데이터 셋
         how={'inner',      # inner join 수행(조인조건에 맞는 데이터만 출력)
              'left',       # left outer join
              'right',      # right outer join
              'outer'},     # full outer join
         on,                # join column
         left_on,           # left data join column
         right_on,          # right data join column
         left_index = False,  # left data index join 여부
         right_index = False, # right data index join 여부
         sort = False)        # 출력결과 정렬 여부

# 1) 컬럼으로 inner join
df2 = DataFrame({'col1':['a','b','c'],
                 'col2':[1,2,3]})

df3 = DataFrame({'col1':['c','b','a'],
                 'col2':[30,20,10]})

pd.merge(df2, df3, on='col1', suffixes=('_df2', '_df3'))
pd.merge(df2, df3, left_on='col1', 
                   right_on='col1', suffixes=('_df2', '_df3'))


# 2) index로 inner join
df22 = df2.set_index('col1')
df33 = df3.set_index('col1')

pd.merge(df22,df33,on='col1')  # index의 이름이 있는 경우 가능

pd.merge(df22,df33,left_index=True, 
                   right_index=True)  # index의 이름이 없는 경우 가능

# 3) outer join
df4 = DataFrame({'col1':['a','b','c','d'],
                 'col2':[1,2,3,4]})
pd.merge(df3, df4, on='col1')                # inner join 수행
pd.merge(df3, df4, on='col1', how='right')   # right outer join 수행

# 3.5 shift 
s2.shift(periods,    # 몇 번째 이전 값을 가져올지
         freq,       # 날짜 오프셋일 경우 날짜 단위 이동 가능
         axis,       # 기본은 이전 행, 컬럼단위로도 이전 값 가져올 수 있음 (axis = 1)
         fill_value) # 이전 값이 없을 경우 NA 대신 리턴 값

# [ 예제 - card_history.csv 파일일 읽고 shift 사용 ]
card = pd.read_csv('card_history.csv', engine='python', encoding = 'euc-kr')
card = card.set_index('NUM')
card = card.applymap(lambda x : x.replace(',','')).astype('int')

card.shift(1, axis=0)  # 행 단위 이동 (이전 값 가져오기)
card.shift(1, axis=1)  # 열 단위 이동

card.shift(-1, axis=0)  # 행 단위 이동(이후 값 가져오기)
# --------------------------------------------------------------------------- # reviewed 1 2020-09-20

# 고급 4. 날짜 변환
from datetime import datetime

# 4.1 strptime    # str(string) p(parsing) time 
# - 문자 -> 날짜
# - datetime 모듈 호출 시 가능
# - 벡터 연산 불가
# - parsing format 생략 불가
d1 = '2020/09/10'
d1.strptime()       # 문자열에 전달(메서드 형식) 불가

datetime.strptime(d1)             # 에러, 2번째 인자 필요
datetime.strptime(d1, '%Y/%m/%d') # 2번째 인자 전달 시 파싱 가능

datetime.strptime('100', '%d')

l1 = ['2020/09/10','2020/09/11','2020/09/12']
datetime.strptime(l1, '%Y/%m/%d')  # 벡터 연산 불가

Series(l1).map(lambda x : datetime.strptime(x, '%Y/%m/%d'))

# 4.2 strftime # str(string) f(format) time 
# - 날짜 -> 문자(날짜의 형식 변경)
# - 메서드, 함수 형식 모두 가능
# - 벡터 연산 불가
t1 = datetime.strptime(d1, '%Y/%m/%d')
t2 = Series(l1).map(lambda x : datetime.strptime(x, '%Y/%m/%d'))

t1.strftime('%A')            # datetime object 적용 가능(메서드 형식)
datetime.strftime(t1, '%A')  # 함수 적용 가능
datetime.strftime(t2, '%A')  # 벡터 연산 불가
# --------------------------------------------------------------------------- #

# 고급 5. 벡터화가 내장된 문자열 메서드 ****
# - pandas 제공
# - 문자열 처리와 관련된 함수 표현식
# - 문자열 input, 문자열 output 
# - upper,
# - str 모듈 호출 lower, replace, find, split ....
# - 벡터연산 가능 후 사용
# - Series만 적용 가능, DataFrame 불가

# 기본 문자열 메서드
# - 기본 함수 
# - 문자열 처리와 관련된 함수 표현식
# - 문자열 input, 문자열 output 
# - upper, lower, replace, find, split ....
# - 벡터연산 불가

L1 = ['a;b;c', 'A;B;C']
s1 = Series(L1)

L1.split(';')                              # 벡터연산 불가

[i.split(';')[0] for i in L1]              # 리스트 내포 표현식 반복 처리
list(map(lambda x : x.split(';')[0], L1))  # mapping 

s1.split(';')       # Series 객체 적용 불가

# 5.1 split
s1.str.split(';')                       # Series 객체 적용 불가
s1.str.split(';')[0]                    # split은 벡터연산 가능, 색인은 불가
s1.str.split(';').map(lambda x :x[0])   # 색인 벡터 처리

# 5.2 replace
s1.replace('a','A')       # 값치환 메서드
s1.str.replace('a','A')   # 벡터화 내장된 문자열 메서드

# =============================================================================
# replace 형태
# 1. 문자열 메서드
# 2. 값치환 메서드
# 3. 벡터화 내장된 문자열 메서드
# =============================================================================

# 5.3 대소치환
s1.upper()      # 불가
s1.str.upper()  # 가능
s1.str.lower()  # 가능
s1.str.title()  # 가능

[ i.title() for i in L1 ] 

# =============================================================================
# 참고 : title의 특징
#
# 'abc'.title()       # 'Abc'
# 'abc ncd'.title()   # 'Abc Ncd'
# 'abc;ncd'.title()   # 'Abc;Ncd'
# 
# =============================================================================

# 5.4 패턴여부
s1.str.startswith('a')    
s1.str.endswith('a')    

s1.str.startswith('a',1)  # position 전달 의미 X

'a' in 'abc'              # 문자열 포함여부는 in 연산자로 처리
'abc'.contains('a')       # 기본 문자열 메서드로는 불가
s1.str.contains('a')      # 문자열의 포함 여부 전달 가능

# 5.5 개수
len('abd')     # 문자열의 크기
len(L1)        # 리스트 원소의 개수(각 원소의 문자열의 크기 X)
s1.str.len()   # 각 원소의 문자열의 크기 리턴

'abcabaa'.count('a')                          # 'a'를 포함하는 횟수
Series(['aa1','abda','a1234']).str.count('a') # 벡터 연산

# 5.6 제거함수(strip)
s2 = Series([' abc ', ' abcd', 'abc12 '])
s2.str.strip().str.len()   # 양쪽 공백 제거 확인
s2.str.lstrip().str.len()  # 왼쪽 공백 제거 확인
s2.str.rstrip().str.len()  # 왼쪽 공백 제거 확인

'abd'.lstrip('a')             # 문자 제거
Series('abd').str.lstrip('a') # 벡터화 내장된 메서드 문자 제거 가능

# 5.7 위치값 리턴(없으면 -1)
'abdd'.find('d')
Series('abdd').str.find('d')

# 5.8 삽입 
a1.pad?     # 문자열 처리 불가

s1.str.pad(width=10,      # 총자리수
           side='both',   # 방향 
           fillchar='-')  # 채울글자

s1.str.pad(10,'both','0')

# 5.9 문자열 결합
'a' + 'b' + 'c' 
Series(['a','A']) + Series(['b','B']) + Series(['c','C']) # 벡터연산 가능

s3 = Series(['ab','AB'])
s3.str.cat(sep=';')       # 결합 기호 전달 가능,
                          # Series의 원소를 결합
s3.str.join(sep=';')      # 결합 기호 전달 가능
                          # 원소별 글자 결합

s1.str.split(';').str.cat(sep='')   # 불가
s1.str.split(';').str.join(sep='')  # Series 매 원소마다 결합

# 5.10 색인
s1.str.split(';').str[0]            # 벡터화 내장된 색인 처리
s1.str.split(';').str.get(0)        # 벡터화 내장된 메서드 처리 가능

# 5.11 중복값 관련 메서드
# 1) Series 적용
t1 = Series([1,1,2,3,4])

t1.duplicated()       # 내부 정렬 후 순차적으로 이전 값과 같은지 여부 확인
t1[t1.duplicated()]   # 중복값 확인
t1[~t1.duplicated()]  # 중복값 제외

t1.drop_duplicates()

# 2) DataFrame 적용
df1 = DataFrame({'col1':[1,1,2,3,4], 
                 'col2':[1,2,3,4,4],
                 'col3':[2,3,4,4,5]})

df1.drop_duplicates('col1', keep='first')         # 첫 번째 값 남김
df1.drop_duplicates('col1', keep='last')          # 두 번째 값 남김
df1.drop_duplicates('col1', keep=False)           # 중복 값 모두 제거

df1.drop_duplicates(['col1','col2'], keep=False)  # 여러 컬럼 전달 가능
# --------------------------------------------------------------------------- #

# 고급 6. group by & cut & 데이터의 결합
# 6.1 group by 기능
# 1) index값이 같은 경우 그룹 연산
# 2) wide 형태(행별, 열별 그룹 연산 가능)
# 3) long 형태(pivot_table)
# 4) group_by

# 6.2 groupby 메서드
# - 분리-적용-결합
# - 특정 컬럼의 값이 같은 경우 grouping(기본 방향)
# - 행별, 컬럼별 grouping 가능
# - tidy 형식(long data)의 데이터에 적용 가능
# - groupby 함수 안에 그룹함수를 전달하는 방식이 X

# 6.2.1 기본
emp = pd.read_csv('emp.csv')

# 1) pivot_table
emp.pivot_table(index='DEPTNO', values='SAL', aggfunc='sum')

# 2) index 생성 후
emp.set_index('DEPTNO')['SAL'].sum(axis=0, level=0)

# 3) groupby
emp.groupby('DEPTNO')              # 분리만 수행
emp.groupby('DEPTNO').sum()        # 연산 가능한 모든 컬럼에 대해 그룹연산
emp.groupby('DEPTNO')['SAL'].sum() # 선택된 컬럼에 대해서만 그룹연산

# 4) 여러 연산 컬럼 전달
emp.groupby('DEPTNO')['SAL','COMM'].mean()

# 5) 여러 groupby 컬럼 전달
emp.groupby(['DEPTNO','JOB'])['SAL'].mean()

# 6) 여러 함수 전달 : agg(결합 함수)
emp.groupby(['DEPTNO','JOB'])['SAL'].agg(['mean','sum'])
emp.groupby(['DEPTNO','JOB'])[['SAL','COMM']].agg({'SAL':'mean',
                                                   'COMM':'sum'})
                                                  
# 예제) emp 데이터에서 deptno별 sal의 평균
emp.groupby('DEPTNO')['SAL'].sum()        # 연산 컬럼 미리 호출 방식
emp['SAL'].groupby(emp['DEPTNO']).sum()   # 연산 컬럼 미리 호출 방식

# 6.2.2 여러 가지 groupby의 옵션
# 1) as_index : groupby 컬럼의 index 전달 여부(기본 : True)
emp.groupby('DEPTNO')['SAL'].sum().reset_index()
emp.groupby('DEPTNO', as_index = False)['SAL'].sum()

# 2) axis(방향 선택), level(multi-index의 depth)
emp2 = emp.sort_values(by = ['DEPTNO','EMPNO']).set_index(['DEPTNO','EMPNO'])

emp2['SAL'].sum(axis=0, level=0)
emp2.groupby(axis=0, level=0)['SAL'].sum()

# 3) 객체를 groupby 컬럼으로 전달
df1 = DataFrame(np.arange(1,17).reshape(4,4),
                index=['a','b','c','d'],
                columns=['A','B','C','D'])

df1.groupby(['g1','g2','g1','g2'], axis=0).sum()
df1.groupby(['g1','g2','g1','g2'], axis=1).sum()

# 4) group_keys : groupby 컬럼의 재출력 방지
#   (groupby 연산 후 연산결과에 groupby 컬럼을 포함하는 경우 생략 가능)

emp.groupby('DEPTNO', group_keys=False)['SAL'].sum()

# 6.3 cut
# - binding 작업
pd.cut(x,                     # 실제 대상(1차원)
       bins,                  # cutting 구간 나열
       right=True,            # 오른쪽 닫힘 여부 (1,2], (2,3], (3,4]
       labels,                # cutting 객체에 이름 부여
       include_lowest=False)  # 최소값 포함 여부

s1 = Series([1,2,3,4,5,6,7,8,9,10])

pd.cut(s1, bins=[1,5,10])     # (1,5], (5,10] 
pd.cut(s1, bins=[1,5,10], labels=('g1','g2'))     
pd.cut(s1, bins=[1,5,10], labels=('g1','g2'), include_lowest=True)

# =============================================================================
# 변수의 변경(binding)
# 1. 연속형 변수를 factor형으로 변경***
# 2. 여러 변수와의 상호작용 고려***
# 
# 학습의 효과
# 시험성적 ~ 학습량X집중력 (interaction)***
# 
# 시험성적 ~ 학습량형태(binding)***
# 
# 학습량(0~10)
# 학습량(11~20)
# 학습량(21~30)
# =============================================================================

# 6.4 데이터의 결합
# 1. append(행 결합)
# 2. merge(컬럼 결합)
# 3. concat(행, 컬럼 결합)
# - 분리되어진 데이터의 union, join 처리
# - 상하결합(axis=0, 기본), 좌우결합(axis=1) 가능
# - join 처리 시 outer join이 기본

df1 = DataFrame({'col1':[1,2,3,4], 'col2':[10,20,30,40]})
df2 = DataFrame({'col1':[1,2,3,4], 'col3':['a','b','c','d']})
df3 = DataFrame({'col1':[5,6], 'col2':[50,60], 'col3':['e','f']})
df4 = DataFrame({'col1':[1,2,3,4], 'col3':['a','b','c','d']},
                index = [0,1,2,4])

df12 = pd.merge(df1, df2, on='col1')
df12.append(df3, ignore_index=True)

pd.concat([df1, df2])          # 세로 방향으로 결합(append 처리, 같은 컬럼끼리)
pd.concat([df1, df2], axis=1)  # 가로 방향으로 결합(index로 join 처리)
                               # merge와는 다르게 중복된 컬럼 생략 X

df12 = pd.concat([df1, df2], axis=1).iloc[:,[0,1,3]]
pd.concat([df12, df3], ignore_index=True)

# df1과 df4를 join
pd.merge(df1, df4, on='col1')
pd.merge(df1, df4, left_index=True, right_index=True) # inner join
pd.concat([df1, df4], axis=1)                         # outer join
# --------------------------------------------------------------------------- #

# 고급 7. 데이터 입출력
# 7.1 read_csv
pd.read_csv(file,        # 파일명
            sep=',',     # 분리구분기호
            header=True, # 첫번째 행 컬럼화 여부, None 설정 시 value로 전달
            names,       # 컬럼이름 변경
            index_col,   # index로 설정할 컬럼이름 전달(multi 가능)***
            usecols,     # 불러올 컬럼 리스트
            dtype,       # 불러올 컬럼의 데이터 타입 지정(딕셔너리 형태)
            engine,   
            skiprows,    # 제외할 행 전달
            nrows,       # 불러올 행 개수 전달
            na_values,   # NA 처리 문자열 전달
            parse_dates, # 날짜 파싱처리할 컬럼 전달 ***
            chunksize,   # 파일을 행 단위로 분리해서 불러올 경우 사용
            encoding)    # 인코딩 옵션

pd.read_csv('read_test.csv', header=None)
pd.read_csv('read_test.csv').dtypes                # date 컬럼이 숫자 형식

pd.read_csv('read_test.csv', parse_dates=['date']) # date 컬럼이 날짜 형식

pd.read_csv('read_test.csv', usecols=['date','a']) # 컬럼 선택 가능

pd.read_csv('read_test.csv', dtype='str')          # 전체 컬럼 데이터 타입 변경
pd.read_csv('read_test.csv', dtype={'c':'str'})    # 특정 컬럼 데이터 타입 변경

pd.read_csv('read_test.csv', index_col = 'date')   # 인덱스 컬럼 지정

pd.read_csv('read_test.csv', na_values=[',','.','!','?','-']) 
pd.read_csv('read_test.csv', na_values={'a' : ['.','-'],
                                        'b' : ['?','!']}) 

pd.read_csv('read_test.csv', names=['date','A','B','C','D']) # header 사라짐

pd.read_csv('read_test.csv', nrows=5)      # 컬럼 제외, 불러올 행의 수
pd.read_csv('read_test.csv', skiprows=5)   # 컬럼 포함, 제외할 행의 수
pd.read_csv('read_test.csv', skiprows=[5]) # 제외할 행 번호 

df_test = pd.read_csv('read_test.csv', chunksize=30) 
df_test            # fetch X

# 7.2 fetch 방법
# 1) 불러올 행의 수 지정, 차례대로 fetch
df_test = pd.read_csv('read_test.csv', chunksize=30) 

df_test1 = df_test.get_chunk(10)
df_test2 = df_test.get_chunk(10)
df_test3 = df_test.get_chunk(10)

# 2) for문을 사용한 print
df_test = pd.read_csv('read_test.csv', chunksize=30) 

for i in df_test :
    print(i)

# 3) for문을 사용하여 하나의 데이터프레임으로 결합***
df_test = pd.read_csv('read_test.csv', chunksize=30)  

df_new = DataFrame()
    
for i in df_test :
    df_new = pd.concat([df_new, i], axis=0)

# 7.3 read_excel
pd.read_excel('emp_1.xlsx', 'Data')

# 7.4 read_clipboard
pd.read_clipboard()
# --------------------------------------------------------------------------- #

# 고급 8. 정규식 패턴
^  : 시작
$  : 끝
.  : 하나의 문자
[] : 여러개 문자 조합 ex) '[0-9]'
\  : 특수기호의 일반기호화
() : group 형성 기호

# 8.1 기본
# replace 메서드와 정규식 표현식의 사용
s1 = Series(['ab12', 'abd!*0', 'abc'])
s2 = Series([1, 2, 3, 4, 5])
s3 = Series(['abcd', 'bcdf', 'Abc'])

# 1) 문자열 메서드 : 사용 불가
'ab12'.replace('[0-9]', '')
'ab[0-9]12'.replace('[0-9]', '')

# 2) 값 치환 메서드
s2.replace('[3-5]', '')                  # 전달 불가
s2.replace('[3-5]', '', regex = True)    # 전달 불가

s3.replace('^a', value = '***')    # 전달 불가
s3.replace(to_replace = '^a', value = '***', regex = True)    # 전달 가능

s2.replace([3, 4, 5], ['a', 'b', 'c'])    # 리스트 대 리스트 매핑 치환 가능

# 3) 벡터화 내장된 (str.replace) 메서드
s1.str.replace('[0-9]', '')
s1.str.replace?

# 8.2 정규식 표현식을 사용한 함수
# 1) findall
# - 정규식 표현식에 매칭되는 값을 추출
# - 벡터 연산 불가
# - str.finall로 벡터 연산 처리 가능 (Series에 적용 가능)
# - 정규식 표현식은 re.compile로 미리 compile 가능 (메모리 절감)
vemail = '''IJ U 12 abc@naver.com 1234 ! abc a123@hanmail.net JHHF\
            ! aa12@daum.net jgjg 3333 ***'''
vemail2 = ['IJ U 12 abc@naver.com 1234 !',
          'abc a123@hanmail.net JHHF',
          '! aa12@daum.net jgjg 3333 ***']

import re              # 정규식 표현식 호출 모듈
re.compile(pattern,    # parsing 할 정규식 표현식
           flags)      # 기타 전달 (대소 구분 같은거 전달)

# 2) 정규식 표현식의 compile
r1 = re.compile('[a-z0-9]+@[a-z]+.[a-z]{1,3}', flags = re.IGNORECASE)   
# + => 1회 이상
# {1, 3} => 1회 이상 3회 이하

# 3) findall로 패턴에 매칭되는 문자열 추출
r1.findall('! aa12@daum.net jgjg 3333 ***')
r1.findall(vemail)              # 문자열의 정규식 표현식 추출 가능
r1.findall(vemail2)             # 벡터 연산 불가
Series(vemail).str.findall(r1)  # Series에서 str.findall로 벡터 연산 가능

pattern1 = '[가-힣]'
r2 = re.compile(pattern1, flags = re.IGNORECASE)
Series(r2.findall('풍납2동')).str.replace(',', '')
# --------------------------------------------------------------------------- #

# 고급 9. 파이썬 날짜 표현
# 날짜 형식 : datetime
from datetime import datetime

dir(datetime)

# 9.1 현재 날짜 출력
d1 = datetime.now()      # datetime.datetime(2020, 9, 17, 9, 19, 12, 526633)

d1.year     # 날짜에서 연 추출
d1.month    # 날짜에서 월 추출
d1.day      # 날짜에서 일 추출
d1.hour     # 날짜에서 시 추출
d1.minute   # 날짜에서 분 추출
d1.second   # 날짜에서 초 추출

# 9.2 날짜 파싱 (문자 -> 날짜)
# 1) datetime.strptime
# - 벡터 연산 불가
# - 2번째 인자 (날짜 포맷) 생략 불가
datetime.strptime('2020/09/17', '%Y/%m/%d')

# 2) pd.to_datetime
# - 벡터 연산 가능
# - 2번째 인자 (날짜 포맷) 생략 가능 (전달 시에는 format = '')
l1 = ['2020/01/01', '2020/09/17']
d2 = pd.to_datetime(l1)
pd.to_datetime(l1, format = '%Y/%m/%d')    # format 인자 이름 생략 불가

# 3) datetime 함수
# - 연, 월, 일, 시, 분, 초 순서대로 값을 전달하여 날짜를 생성 (파싱)
# - 벡터 연산 불가
vyear = [2007, 2008]
vmonth = [7, 8]
vday = [7, 8]

datetime(2007, 9, 11)            # 각 인자에 연, 월, 일 순서로 정수 전달
datetime(vyear, vmonth, vday)    # 각 인자에 리스트 전달 불가, 벡터 연산 불가

# 9.3 포맷 변경
datetime.strftime(d1, '%A')
datetime.strftime(d2, '%A')    # d2의 전달 불가, 벡터 연산 불가

d1.strftime('%A')
d2.strftime('%A')              # d2(datetimeindex) 전달 가능, 벡터 연산 가능 / Series 객체 전달 불가

# 9.4 날짜 연산
t1 = datetime(2020, 9, 17)
t2 = datetime(2020, 9, 10)

t1 - t2            # 날짜 - 날짜 연산 가능 (기본 단위 : 일), timedelta object
(t1 - t2).days     # timedelta object의 일 수 출력
(t1 - t2).seconds  # timedelta object의 초 수 출력

# 1) timedelta를 사용한 날짜 연산
from datetime import timedelta

d1 + 100                          # 날짜 + 숫자 연산 불가
d1 + timedelta(days = 100)        # 100일 뒤

# 2) offset으로 사용한 날짜 연산
import pandas.tseries.offsets
dir(pandas.tseries.offsets)

from pandas.tseries.offsets import Day, Hour, Second

Day(5)      # 5일
Hour(5)     # 5시간
Second(5)   # 5초 

d1 + Day(100)

# 9.5 날짜 인덱스 생성 및 색인
# pd.date_range : 연속적 날짜 출력
pd.date_range(start,             # 시작 날짜
              end,               # 끝 날짜
              periods,           # 기간 (출력 개수)
              freq)              # 날짜 빈도 (매월, 매주 ...)

pd.date_range(start = '2020/01/01', end = '2020/01/31')    # 기본 freq = 'D'(일)
pd.date_range(start = '2020/01/01', periods = 100)     # 시작값으로부터 100일의 날짜

pd.date_range(start = '2020/01/01', end = '2020/01/31',
              freq = '7D')    # by값과 비슷

# [ 참고 - freq의 전달 ]
pd.date_range(start = '2020/01/01', end = '2020/01/31', freq = 'D')     # 일
pd.date_range(start = '2020/01/01', end = '2020/01/31', freq = '7D')    # 7일
pd.date_range(start = '2020/01/01', end = '2020/01/31', freq = 'W')     # 매주 일
pd.date_range(start = '2020/01/01', end = '2020/01/31', freq = 'W-WED') # 매주 수
pd.date_range(start = '2020/01/01', end = '2020/01/31', freq = 'W-MON') # 매주 월
pd.date_range(start = '2020/01/01', end = '2020/12/31', freq = 'MS')    # 매월 1일
pd.date_range(start = '2020/01/01', end = '2020/12/31', freq = 'M')     # 매월 말일
pd.date_range(start = '2020/01/01', end = '2020/12/31', freq = 'BMS')   # BusinessMonthBegin, 매월 첫 영업일
pd.date_range(start = '2020/01/01', end = '2020/12/31', freq = 'BM')    # BusinessMonth, 매월 마지막 영업일
pd.date_range(start = '2020/01/01', end = '2020/12/31', freq = 'WOM-3FRI') # WeekOfMonth, 매월 셋째주 금요일

# 날짜 인덱스를 갖는 Series 생성
date1 = pd.date_range('2020/01/01', '2020/03/31')
s1 = Series(np.arange(1, len(date1) + 1), index = date1)

# 날짜 인덱스의 색인
s1['2020']    # 날짜에서 특정 년에 대한 색인 가능
s1['2020-03'] # 날짜에서 특정 년/월에 대한 색인 가능

# truncate : 날짜 인덱스를 갖는 경우의 날짜 선택 메서드
s1.truncate(after = '2020-03-23')    # 처음 ~ 2020-03-23 까지 출력
s1.truncate(before = '2020-03-23')   # 2020-03-23 ~ 끝 까지 출력

# 날짜 슬라이스
s1['2020-03-23':'2020-03-27']    # 끝 범위 포함

# 9.6 resample : 날짜의 빈도수 변경
# - 서로 다른 offset을 갖는 Series나 DataFrame의 연산 시
# - offset을 맞춰놓고 연산 시 사용
# - 같은 offset끼리 그룹연산 가능 (downsampling 경우)
# - upsampling : 더 많은 날짜수로 변경 (주별 -> 일별)
# - downsampling : 더 적은 날짜수로 변경 (일별 -> 월별)

# 1) downsampling 예제 : 일별 데이터를 주별 데이터로 변경
s1.resample(rule,        # 날짜 빈도
            axis = 0)    # 방향 (0: index의 resample, 1: column의 resample)
            
s1.resample('M', how = 'sum')    # (구) downsampling의 경우 how로 그룹함수를 전달할 수 있었음

s1.resample('M').sum()           # downsampling의 경우 그룹함수를 추가 전달
                                 # M은 MonthEnd를 의미하므로 매월 마지막 날짜 리턴
                                 # 월별 grouping 기능을 가지고 있음
                                 
# 2) upsampling 예제 : 주별 데이터를 일별 데이터로 변경
date2 = pd.date_range('2020/01/01', '2020/03/31', freq = '7D')
d3 = Series(np.arange(10, len(date2) * 10 + 1, 10), index = date2)

d3.resample('D')    # upsampling의 경우 자동으로 새로 생긴 날짜 생성 x

d3.resample('D', fill_method = 'ffill')    # (구) fill_method로 이전 날짜 값 가져올 수 있었음

d3.resample('D').sum()    # 새로 생긴 값(NA)을 0으로 치환 => NA 합하면 0이 출력되는 성질 이용
d3.resample('D').asfreq() # 새로 생긴 값을 NA로 리턴

d3.resample('D').asfreq().fillna(0)                  # 새로 생긴 값(NA) 0으로 리턴
d3.resample('D').asfreq().fillna(method = 'ffill')   # fill_method = 'ffill'

d3.resample('D').ffill()                             # fill_method = 'ffill'

# multi-index를 갖는 경우 resample 전달
df3 = d3.reset_index()
df3.columns = ['date', 'cnt']
df3.index = [['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B'], df3.date]

df3.resample('D').asfreq()                 # Error => MultiIndex의 레벨 전달 필요
df3.resample('D', level = 'date').sum()    # MultiIndex의 레벨 전달 시 필요
df3.resample('D', on = 'date').sum()       # 날짜컬럼 지정 시 필요

# [ 참고 ]
# asfreq는 resample의 level 혹은 on 인자 사용 시 전달 불가
# multi index의 특정 레벨로 resample하는 경우 다른 레벨 값 생략됨

# [ 참고 : dropna ]
# - NA값을 갖는 행(기본), 컬럼 제거 (axis로 선택 가능)
# - NA가 하나라도 포함된 행, 컬럼 제거 (조절 가능)

df_na = DataFrame(np.arange(1, 26).reshape(5, 5))
df_na.iloc[1, 0] = NA
df_na.iloc[2, [0, 1]] = NA
df_na.iloc[3, [0, 1, 2]] = NA
df_na.iloc[4, [0, 1, 2, 3, 4]] = NA

df_na.dropna()               # axis = 0, NA가 하나라도 포함된 행을 모두 제거
df_na.dropna(how = 'any')    # NA가 하나라도 포함된 행을 모두 제거

df_na.dropna(how = 'all')    # 전체 행이 NA인 경우만 삭제
df_na.dropna(thresh = 1)     # NA가 아닌 값이 1개 이상을 리턴
df_na.dropna(thresh = 3)     # NA가 아닌 값이 3개 이상을 리턴

df_na.dropna(axis = 1, thresh = 3)    # 컬럼 삭제, NA가 아닌 값이 3개 이상을 리턴

# [ 참고 - 사용자 정의 함수 생성, NA 개수 기반 삭제 ]
pd.isnull(df_na.iloc[4, :]).sum() == 5
f_dropna = lambda x, n = 1 : pd.isnull(x).sum() >= n
df_na.loc[~df_na.apply(f_dropna, axis = 1, n = 3), :]
# --------------------------------------------------------------------------- #

# 고급 10. 파이썬 시각화

# 10.1 figure와 subplot
# - figure : 그래프가 그려질 전체 창 (도화지 개념)
# - subplot : 실제 그림이 그려질 공간 (분할 영역)
# - in R : par(mfrow = c(1, 3)) => 하나의 도화지에 3개의 분할 영역
# - figure와 subplot에 이름 부여 가능
# - 기본적으로 하나의 figure와 하나의 subplot이 생성

# [ 참고 : 각 창에서 시각화모드(pylab) 전환 방법 ]
# 1) cmd
# 1-1) anaconda prompt(ipython)
# ipython           # cmd에서 실행
# $matplotlib qt    # anaconda prompt

# 1-2) pylab 아나콘드 모드 직접 전환 
# ipython -- pylab           # cmd에서 실행

# 2) spyder tool
# Preferences > Ipython Console > Graphics > Graphics Backend > Automatic > 재실행

# 2-1) figure와 subplot 생성
run profile1
import matplotlib.pyplot as plt

fig1 = plt.figure()    # figure 생성
ax1 = fig1.add_subplot(2,    # figure 분할 행의 수
                       2,    # figure 분할 컬럼의 수
                       1)    # 분할된 subplot의 위치 (1부터 시작)

ax2 = fig1.add_subplot(2,    # figure 분할 행의 수
                       2,    # figure 분할 컬럼의 수
                       1) 

s1 = Series([1, 10, 2, 25, 4, 3])
ax2.plot(s1)    # ax2 subplot에 직접 plot 도표 전달 => 특정 그래프에 데이터 전달 가능 (순서 상관없이 위치전달 가능)

# 2-2) figure와 subplot 동시 생성
# - plt.subplots로 하나의 figure와 여러 개의 subplot 동시 생성 
# - 이름 부여 시 figure 이름과 subplot 대표 이름 각각 지정
# - subplot 위치 지정은 색인
plt.subplots(nrows = 1,          # figure 분할 행의 수
             ncols = 1,          # figure 분할 컬럼의 수
             sharex = False,     # 분할된 subplot의 x축 공유 여부
             sharey = False)     # 분할된 subplot의 y축 공유 여부

fig2, ax = plt.subplots(2, 2)
ax[1, 1].plot(s1)          

# 10.2 선 그래프 그리기
# 1) Series 전달
ax[0, 1].plot(s1)        # 특정 figure, subplot에 그리는 방법
s1.plot()                # 가장 마지막 figure 혹은 subplot에 전달, 새로 생성

# 2) DataFrame 전달
# - 컬럼별 서로 다른 선 그래프 출력
# - 컬럼이름값이 자동으로 범례 생성 (위치 가장 좋은 자리)
# - 인덱스이름값이 자동으로 x축 생성

# [ 예제 - 선그래프 그리기 ]
# - fruits.csv 파일을 읽고 과일별 판매량 증감 추이 시각화
fruits = pd.read_csv('fruits.csv')
fruits2 = fruits.pivot('year', 'name', 'qty')

fruits2.plot()

# 10.2.1 선 그래프 옵션 전달
# 1) plot 메서드 내부 옵션 전달 방식 (상세 옵션 전달 불가)
# legend의 위치, 글씨 옵션
fruits2.plot(xticks,        # x축 눈금
             ylim,          # y축 범위
             fontsize,      # 글자 크기
             rot,           # (x축 이름) 글자 회전방향
             color,         # 선 색
             linestyle,     # 선 스타일
             marker,        # 선 모양
             style,         # (종합) 선 스타일
             title,         # 그래프 이름
             kind)          # 그래프 종류 (default = 선 그래프)
             
fruits2.plot(xticks = fruits2.index,
             style = '--')             
             
# [ 참고 : 선 스타일 종류 및 전달 ]
# 'r--' : 붉은 대시선
# 'k-'  : 검은색 실선
# 'b.'  : 파란색 점선
# 'ko--': 검은색 점모양 대시선

s1.plot(style = 'b.')
s1.plot(color = 'b', linestyle = '--', marker = 'o')

s1.index = ['월', '화', '수', '목', '금', '토']
s1.plot()    # x축 이름이 깨짐 (한글)

plt.rc('font', family = 'AppleGothic')    # 글씨체 변경방법

# 2) 광 범위 옵션 전달 방식 : plt.옵션함수명
# 종류 확인 : dir(plt)
# 2-1) x축, y축 이름
plt.xlabel('발생년도')
plt.ylabel('검거율')

# 2-2) 그래프 제목
plt.title('구별 년도별 검거율 변화')

# 2-3) x축, y축 범위
plt.ylim([0, 130])

# 2-4) x축, y축 눈금
plt.xticks(cctv2.index)

# 2-5) legend ***
plt.legend(fontsize = 6,
           loc = 'upper right',
           title = '구 이름')

# [ 참고 : 선그래프 옵션 확인 방법 ]
plt.plot(s1)    # 선 스타일, 마커 스타일 확인 가능

# 10.3 barplot 그리기
# - 행별로 서로 다른 그룹
# - 각 컬럼의 데이터들이 서로 다른 막대로 출력 기본 (R에서는 beside = T)
# - 각 컬럼의 데이터들이 하나의 막대로 출력(stacked = True)
# - 컬럼 이름이 자동으로 범례 이름으로 전달
# - 인덱스 이름이 자동으로 x축 이름으로 전달

fruits2.plot(kind = 'bar')
plt.xticks(rotation = 0)    # x축 눈금 회전

run profile1
df1=pd.read_csv('card_history.csv', encoding='euc-kr', engine='python')
f1=lambda x : x.str.replace('.', '').astype('int')
df1.iloc[:, 1:].apply(f1, 0).sum(0)

# [ 참고 : plt.rc로 global 옵션 전달 방식 ]
plt.rcParams.keys()    # 각 옵션그룹별 세부 옵션 정리

plt.rc(group,        # 파라미터 그룹
       **kwargs)     # 상세 옵션
       
plt.rc('font', family = 'AppleGothic')       

# 10.4 히스토그램
cctv['CCTV수'].hist(bins=10)                     # 막대의 개수
cctv['CCTV수'].hist(bins=[0,100,200,500,1000])   # 막대의 범위

cctv['CCTV수'].plot(kind='kde')                  # 커널 밀도 함수(누적분포)