# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 28.
# 28.1 card_history.txt 파일을 array로 읽고 (첫번째 행 제외)
card1 = np.loadtxt('card_history.txt', dtype = 'str', delimiter = '\t', skiprows = 1)

# 1) 각 품목별 총합 출력                                                           # *
    # 2차원 천단위 구분기호 제거
    # way 1) 함수의 input이 스칼라인 경우
    a1 = card1[:, [1, 2, 3, 4, 5, 6]]
    f1 = lambda x : int(x.replace(',', ''))    # 스칼라가 들어오므로 int처리 (not astype)
    list(map(f1, card1)])    # Error => card1[0], card1[1], ... 단위로 들어가기 때문 *
    
    list(map(f1, card1[:, 0]))  
    
    # 컬럼 개수가 적다면 1개씩 개별 진행
    np.array(list(map(f1, a1[:, 0]))).sum()
    np.array(list(map(f1, a1[:, 1]))).sum()
    np.array(list(map(f1, a1[:, 2]))).sum()
    np.array(list(map(f1, a1[:, 3]))).sum()
    np.array(list(map(f1, a1[:, 4]))).sum()
    np.array(list(map(f1, a1[:, 5]))).sum()
    
    # way 2) 함수의 input이 2차원 array *
    def f2(x) :
        outlist = []
        for i in x :
            inlist = []
            for j in i :
                inlist.append(int(j.replace(',', '')))
            outlist.append(inlist)
        return np.array(outlist)
    
    card1_1 = f2(card1)
    
    card1_1[:, 1:].sum(axis = 0)
    
    # way 3) 함수의 input이 스칼라인 경우 + applymap => easy & best    
    DataFrame(card1).applymap(f1)    # 아래 참고 참조
    
    DataFrame(card1_1[:, 1:]).apply(sum, axis = 0)

# 2) 의복 지출이 가장 많은 날의 총 지출 금액 출력
card1_1[np.argmax(card1_1[:, 2]), :][1:].sum()

# 28.2 student.csv 파일을 읽고
std = np.loadtxt('student.csv', dtype = 'str', delimiter = ',', skiprows = 1, encoding = 'euc-kr')

# 1) id에 'o'가 포함된 학생의 이름, 학년, id 출력
std[np.ix_(['o' in x for x in std[:, 2]], [1, 4, 3])]

# 2) 성별을 나타내는 array 생성
np.where(std[:, 4].startswith('2', 7), 'Female', 'Male')
gender = np.array([ x.startswith('2', 7) for x in std[:, 4]])
gender = np.where(gender == 1, 'Female', 'Male')

# 3) vdeptno라는 변수 생성, deptno2가 있으면 해당 값을, 없으면 deptno1 값 리턴
vdeptno = np.where(std[:, -2] == '', std[:, -3], std[:, -2])

# 4) 일지매를 이윤정으로 수정 (위치 색인 불가)
vname = std[std[:, 1] == '일지매', 1]
vname = '이윤정'

# 5) 서재수와 김주현의 이름, 학년, 전화번호 출력
std[np.ix_(np.in1d(std[:, 1], ['서재수', '김주현']), [1, 3, 6])]
# --------------------------------------------------------------------------- #

# -------------------------------- 실 습 문 제 -------------------------------- ## *
# 실습문제 6.
# 6.1 test3 파일을 불러온 후 소수점 둘째자리로 표현
# 1) Array
test1 = np.loadtxt('test3.txt', delimiter = '\t')

def f1(x) :
    outlist = []
    for i in x :
        inlist = []
        for j in i :
            inlist.append('%.2f' % float(j))
        outlist.append(inlist)
    return np.array(outlist)
            
f1(test1)        
    
# 2) DataFrame
test2 = pd.read_csv('test3.txt', sep = '\t', header = None)

'%.2f' % 3
f2 = lambda x : '%.2f' % x

list(map(f2, test1))        # 2차원 형식 전달 불가
list(map(f2, test1[0]))     # 1차원 형식 전달 가능

test1.applymap(f2)               # numpy에 적용 불가
DataFrame(test1).applymap(f2)    # pandas(DataFrame)에 적용 가능

# 6.2 read_test 파일을 읽고
test2 = pd.read_csv('read_test.csv')
test2.dtypes    # a, b 컬럼이 문자열로 들어옴

test3 = pd.read_csv('read_test.csv', na_values = ['?', '.', '-', '!'])
test3.dtypes

# 1) .,-,?,!,null,nan 값을 모두 0으로 수정
# sol1) 각 컬럼별 치환(np.where)
test2.a = np.where((test2.a == '.') | (test2.a == '-'), 0, test2.a)

# sol2) 각 컬럼별 치환(map)
f3 = lambda x : 0 if x in ['.', '-'] else x

list(map(f3, test2.a))    # 결과가 리스트로 나와서 불편
test2.a.map(f3)

# sol3) 데이터 프레임 전체 적용
    # test1
    def f4(x) :
        if x in ['.', '-', '?', '!'] | np.isnan(x):
            return 0
        else:
            return x
        
    test2.applymap(f4)    # Error => np.isnan 함수가 문자열에 대한 input 허용하지 않음 ex) '.', '-'
    
    # test2
    pd.isnull(0)
    pd.isnull('0')    # 문자열 input 허용
    pd.isnull(NA)
    
    def f4(x) :
        if x in ['.', '-', '?', '!'] | pd.isnull(x):
            return 0
        else:
            return x
        
    test2.applymap(f4)    # Error => np.isnan 함수가 문자열에 대한 input 허용하지 않음 ex) '.', '-'    

np.isnan(NA)
def f4(x) :
    if x in ['.', '-', '?', '!']:
        return 0
    else:
        return x
    
test2 = test2.applymap(f4)    

def f5(x) :
    if np.isnan(float(x)):
        return 0
    else:
        return x

test2 = test2.applymap(f5)

# =============================================================================
# [ 참고 - Python에서의 NA 체크 ]
# np.isnan('a')     # np.isnan은 문자열에 대한 NA 확인 불가
# np.isnan(NA)
# 
# pd.isnull('a')    # pd.isnull은 문자열에 대한 NA 확인 가능
# pd.isnull(NA)
# pd.isnull(Series([1, 2, NA]))    # 벡터연산 가능
# =============================================================================

# 2) 컬럼별 총합 출력
# step 1) 데이터프레임 각 컬럼 숫자로 변경
# 문자 컬럼 -> 숫자 컬럼으로 변경
test2.astype('int')                # 형변환 메서드(astype)는 2차원 데이터 셋 벡터 연산 가능
int(test2)                         # 형변환 함수는 벡터 연산 불가
test2 = test2.applymap(lambda x : int(x))  # applymap으로 2차원 데이터셋에 적용 

# step 2) 컬럼별 총합 계산
test2.dtypes
test2.apply(sum, axis = 0)

# 3) year, month, day 컬럼 생성
test2.date

test2.date.astype('str')[:4]    # 원하는 결과가 아님 => 4개의 스칼라 가져옴

f6 = lambda x : str(x)[:4]
f7 = lambda x : str(x)[4:6]
f8 = lambda x : str(x)[6:8]

test2.year = test2.date.map(f6)
test2.month = test2.date.map(f7)
test2.day = test2.date.map(f8)

# 4) d값이 가장 높은 날짜 확인
test2.date[np.argmax(test2.d)]

# 3. crime.csv 파일을 읽고
test3 = pd.read_csv('crime.csv', encoding = 'euc-kr')

# 1) 검거/발생*100 값을 계산 후 rate 컬럼에 추가
test3.rate = test3.검거 / test3.발생 * 100

# =============================================================================
# 데이터프레임의 컬럼 생성 방법
# df1.column_name = 값
# =============================================================================

# --------------------------------------------------------------------------- #

# -------------------------------- 실 습 문 제 -------------------------------- #
# 실습문제 7.
# 7.1 아래와 같은 데이터 프레임 생성 후 (세 개의 컬럼을 갖는)
# name price qty
# apple 2000 5
# mango 1500 4
# banana 500 10
# cherry 400 NA
d1 = {'name' : ['apple', 'mango', 'banana', 'cherry'],
      'price' : [2000, 1500, 500, 400],
      'qty' : [5, 4, 10, NA]}
df1 = DataFrame(d1)

# =============================================================================
# # Python 색인 방식
# # 1. 기본 색인 R과 유사 : [row index, column index]
# # 2. numpy의 리스트 색인 시 : [[row index], [column index]] 불가
#      => np.ix_([row index], [column index])
# # 3. pandas 색인 : [row index, column index] 불가
#      => df1.iloc[row index, column index]    # 위치기반 색인
#      => df1.loc[row index, column index]     # 이름기반 색인 (조건 가능)
# =============================================================================
     
# 1) mango의 price와 qty 선택
df1.loc[1, ['price', 'qty']]    # Positional로 색인하면 차원축소 되어서 시리즈로 나옴
df1.name
df1['name']

df1[df1.name == 'mango', ['price', 'qty']]    # Error
df1.loc[df1.name == 'mango', ['price', 'qty']]    # 차원축소 안됨

# 2) mango와 cherry의 price 선택                                                 # *
df1.loc[[1, 3], 'price']
df1.iloc[[1, 3], 1]

df1.loc[(df1['name'] == 'mango') | (df1['name'] == 'cherry'), 'price']
df1['name'] in ['mango', 'cherry']    # Error => 벡터연산 불가

# =============================================================================
# # pandas에서 in 연산자 : isin**
# df1['name'][0] in ['mango', 'cherry']    # 스칼라에 대해 in 연산자 처리 가능
# df1['name'] in ['mango', 'cherry']       # 시리즈에 대해 in 연산자 벡터연산 불가
# 
# df1['name'].map(lambda x : x in ['mango', 'cherry'])
# df1['name'].isin(['mango', 'cherry'])    # 위 연산과 같음 => 하나씩 진행
# =============================================================================

# 3) 전체 과일의 price만 선택
df1.loc[:, 'price']    # 시리즈 출력
df1.iloc[:, 1:2]       # DF 출력
df1.loc[:, 'price':'price']    # DF 출력
df1.loc[:, 'price':'qty']      # DF 출력 => loc의 경우 숫자 색인과 다름

# =============================================================================
# # [ 참고 - 슬라이스 색인의 형태 ]
# n:m => n에서 (m-1)까지 추출
# name1:name2 => name1에서 name2까지 추출 (마지막 범위 포함))
# =============================================================================

# 4) qty의 평균
df1.qty.mean()    # 자동으로 NA는 제외
np.mean(df1.qty)

df1_1 = df1.loc[:, 'qty']
df1_1.loc[-pd.isnull(df1_1)].mean()

# 5) price가 1000 이상인 과일 이름 출력
df1.loc[df1.price >= 1000, 'name']           # 시리즈 출력
df1.loc[df1.price >= 1000, 'name':'name']    # DF 출력

# 6) cherry, banana, mango, apple 순 출력                                       # *
# 위치값 기반
df1.iloc[[3, 2, 1, 0]]

# 이름 기반 reindexing
df1.index = df1.name
df1 = df1.iloc[:, 1:]    # name 제외

df1.loc[['cherry', 'banana', 'mango', 'apple']]
DataFrame(df1, index = ['cherry', 'banana', 'mango', 'apple'])
df1.reindex(['cherry', 'banana', 'mango', 'apple'])

# =============================================================================
# # DataFrame
# # 1. 생성
# # 2. 색인
# # 3. 기본 메서드
# df1.index
# df1.columns
# 
# df1.values        # key값 제외한 순수 데이터
# df1.index.name    # index의 이름
# df1.columns.name  # index의 이름
# =============================================================================

# 7.2 emp.csv 파일을 읽고
emp = pd.read_csv('emp.csv')
df2 = pd.read_csv('emp.csv')

# 1) 이름 컬럼 값을 모두 소문자로 변경
'ALLEN'.lower()

df2.ENAME.lower()  # 벡터연산 불가
df2.ENAME = df2.ENAME.map(lambda x : x.lower())

# 2) 아래 인상 규칙대로 인상된 연봉을 계산 후 new_sal 컬럼에 추가
# (컬럼 추가 방식 : df1['new_sal'] = values)
# 10번 부서는 10%, 20번 부서는 15%, 30번 부서는 20%
# Answer 1 => for문
for i, j in zip(df2.DEPTNO, df2.SAL) :
    print('부서번호 : %s, 연봉 : %s' % (i, j))

vsal = []
for i, j in zip(df2.DEPTNO, df2.SAL) :
    if i == 10 :
        vsal.append(j * 1.1)
    elif i == 20 :
        vsal.append(j * 1.15)
    else :
        vsal.append(j * 1.2)
        
df2['new_sal'] = vsal     

# Answer 2 => np.where
vsal2 = np.where(df2.DEPTNO == 10, df2.SAL * 1.1,
         np.where(df2.DEPTNO == 20, df2.SAL * 1.15, df2.SAL * 1.2))
df2['new_sal2'] = vsal2

# Answer 3 => mapping 처리
f1 = lambda x : '인사부' if x == 10 else '총무부'
df2.DEPTNO.map(f1) 

def f2(x, y) :
    if x == 10 :
        return(y * 1.1)
    elif x == 20 :
        return(y * 1.15)
    else :
        return(y * 1.2)

f2(10, 800)                    # 스칼라 전달 가능
f2(df2.DEPTNO, df2.SAL)        # 벡터연산 불가

df2.DEPTNO.map(f2, df2.SAL)    # map 메서드는 추가 인자 전달 불가
l1 = list(map(f2, df2.DEPTNO, df2.SAL))   # map 함수는 추가 인자 전달 가능

df2['new_sal3'] = l1

# [ 위 for문 사용 시 주의 ]
df2.loc[0, 'ENAME']
for i in range(0, 14) :
    if df2.loc[i, 'DEPTNO'] == 10 :
        df2['new_sal2'][i] = df2.loc[i, 'SAL'] * 1.1
    elif i == 20 :
        df2['new_sal2'][i] = df2.loc[i, 'SAL'] * 1.15
    else :
        df2['new_sal2'][i] = df2.loc[i, 'SAL'] * 1.2
        
# Error => 없는 키에 값 입력 불가
df2['new_sal2'][0] = 10    # R에서만 가능한 표현식

# out of range 상황 해결하기 위해 리스트의 전체 값 임의 할당
l1 = list(np.arange(1, 15))
for i in range(0, 14) :
    if df2.loc[i, 'DEPTNO'] == 10 :
        l1[i] = df2.loc[i, 'SAL'] * 1.1
    elif i == 20 :
        l1[i] = df2.loc[i, 'SAL'] * 1.15
    else :
        l1[i] = df2.loc[i, 'SAL'] * 1.2

# 3) comm이 없는 직원은 100 부여
# Answer 1 => np.where
np.where(pd.isnull(df2.COMM), 100, df2.COMM)

# Answer 2 => map
df2.COMM.map(lambda x : 100 if pd.isnull(x) else x)

# Answer 3
emp.loc[pd.isnull(emp.COMM), 'COMM'] = 100

# -------------------------------- 실 습 문 제 -------------------------------- #
# 실습문제 8.
# 8.1 'test3.txt' 파일을 읽고 
np1 = np.loadtxt('test3.txt')
df1 = DataFrame(np1, index = ind, columns = col)
'2000' + '년'
a1 = np.arange(2000, 2014)
a2 = np.arange(20, 70, 10)
ind = [str(x) + '년' for x in a1]
col = [str(x) + '세이상' if x == 60 else str(x) + '대' for x in a2]


# 1) 다음과 같은 데이터 프레임 형태로 변경
# 	     20대	30대 40대 50대 60세이상
# 2000년	  7.5	3.6	 3.5  3.3	1.5
# 2001년	  7.4	3.2	 3	  2.8	1.2
# 2002년	  6.6	2.9	 2	  2	    1.1
# ..............................................
# 2011년	  7.4	3.4	 2.1  2.1	2.7
# 2012년	  7.5	3	 2.1  2.1	2.5
# 2013년	  7.9	3	 2	  1.9	1.9

# 2) 2010년부터의 20~40대 실업률만 추출하여 새로운 데이터프레임 생성
df1_1 = df1.iloc[10:, 0:3]

# 3) 30대 실업률을 추출하되, 소수점 둘째자리의 표현식으로 출력
f1 = lambda x : '%.2f' % x
df1.iloc[:, 1].map(f1)

# 4) 60세 이상 컬럼 제외
df1 = df1.iloc[:, 0:4]

# 5) 30대 컬럼의 값이 높은순 정렬

# 8.2 subway2.csv  파일을 읽고                                                    # * 정답 체크
subway2 = pd.read_csv('subway2.csv', encoding = 'euc-kr', skiprows = 1)
# 1) 다음의 데이터 프레임 형식으로 변경
# 전체     구분   5시       6시     7시 ...
# 서울역  승차 17465  18434  50313 ...
# 서울역  하차 ....
subway2.전체 = subway2.전체.fillna(method = 'ffill').str.split('(').str[0]

# 2) 각 역별 하차의 총합
subway2.loc[subway2.구분 == '하차', '05~06':'24~01'].sum(axis = 1)

# 3) 승차의 시간대별 총합
subway2
subway2.loc[subway2.구분 == '승차', '05~06':'24~01'].sum(axis = 0)


# 4) 하차 인원의 시간대별 각 역의 차지 비율 출력
subway2.loc[subway2.구분 == '하차', '05~06':'24~01'] / subway2.loc[subway2.구분 == '승차', '05~06':'24~01'].sum(axis = 0) * 100

# --------------------------------------------------------------------------- # reviewed 1, 2020-09-21

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 33.
# multi_index.csv 파일을 읽고 멀티 인덱스를 갖는 데이터프레임 변경
df33 = pd.read_csv('multi_index.csv', engine = 'python', encoding = 'euc-kr')
df1 = pd.read_csv('multi_index.csv', engine = 'python', encoding = 'euc-kr')

# Answer 1
# step 1) 첫번째 컬럼(지역) NA 치환 (이전값)
df33.iloc[:, 0] = df33.iloc[:, 0].fillna(method = 'ffill')

# step 2) 멀티 인덱스 설정
    # sol1) 
    df33.index = [df33.iloc[:, 0], df33.iloc[:, 1]]
    # sol2)
    df33.set_index(['Unnamed :  0', 'Unnamed :  1'])

# step 3) 멀티 인덱스 이름 변경
df33.index.names = ['지역', '지점']

# step 4) 컬럼 이름 변경 (냉장고, 냉장고, TV, TV)
    # sol1) 직접 수정
    df33.columns = ['냉장고', '냉장고', 'TV', 'TV']
    
    # sol2) Unnamed를 포함한 값을 NA로 수정 후 이전 값 치환
    'Unnamed' in 'Unnamed: 3'
    col1 = df33.columns.map(lambda x : NA if 'Unnamed' in x else x)
    col1.fillna(method = 'ffill')    # Error => index obejct에서는 method 옵션 불가
    
    df33.columns = Series(col1).fillna(method = 'ffill')  # Series로 변경 후 처리
    
# step 5) 멀티 컬럼 설정
# [현재컬럼, 1번째행]
df33.columns = [df33.columns, df33.iloc[0, :]]

# step 6) 1번째 행 제거
    # sol1)
    df33 = df33.iloc[1:, ]
    # sol2) 
    df33.drop(NA, axis = 0, level = 0)    # 멀티 인덱스일 경우 레벨 전달 필요

# step 7) 멀티 컬럼 이름 설정
df33.columns.names = ['품목', '구분']

# Answer 2
df1.index = [['제거', 'seoul', 'seoul', 'incheon', 'incheon'], ['제거', 'A', 'B', 'A', 'B']]
df1.columns = [['하위레벨', '하위레벨', '냉장고', '냉장고', 'TV', 'TV'],
               ['상위레벨', '상위레벨', 'price', 'qty', 'price', 'qty']]
df1 = df1[1:5, 0:6]
df1 = df1.drop('제거', axis = 0)
df1 = df1.iloc[:, 2:]
df1.index.names
df1.columns
# --------------------------------------------------------------------------- #

# -------------------------------- 실 습 문 제 -------------------------------- #
# 실습문제 9.
# 9.1 card_history.txt 파일을 읽고
run profile1
card = pd.read_csv('card_history.txt', sep='\s+')                              # *
# sep = '\t'  : 탭 분리구분(공백으로 분리된 컬럼을 분리하지 X)         
# sep = '\s+' : 한 칸의 공백 이상(탭으로 분리된 컬럼도 분리 가능)

# 인덱스 설정   
# 천단위 구분기호 제거 후 숫자 변경
card.replace(',','')     # 값 치환 메서드이므로 ',' 패턴 치환 X
'19,400'.replace(',','') # 치환 가능

card.iloc[:, 1:] = card.iloc[:, 1:].applymap(lambda x : int(x.replace(',','')))

# 1) 각 일별 지출품목의 차지 비율 출력(식료품 : 20%, 의복 : 45%, ....)                   # *
card.iloc[0,:] / card.iloc[0,:].sum() * 100
card.iloc[1,:] / card.iloc[1,:].sum() * 100
card.iloc[2,:] / card.iloc[2,:].sum() * 100

f1 = lambda x : x / x.sum() * 100
card.apply(f1, axis=1)

# 2) 각 지출품목별 일의 차지 비율 출력(1일 : 0.7%, 2일 : 1.1%, ....)
card.apply(f1, axis=0)

# 3) 각 일별 지출비용이 가장 높은 품목 출력                                             # *        
# sol1) argmax 사용
card.columns[card.iloc[0,:].argmax()]
card.columns[card.iloc[1,:].argmax()]
card.columns[card.iloc[2,:].argmax()]

f2 = lambda x : card.columns[x.argmax()] 
card.apply(f2, axis=1)

# sol2) idxmax 사용
card.iloc[0,:].idxmax()
card.iloc[1,:].idxmax()
card.iloc[2,:].idxmax()

f3 = lambda x : x.idxmax()

card.apply(f3, axis=1)     # apply를 사용한 각 행별 적용
card.idxmax(axis=1)        # idxmax 자체 행별 적용(axis)

# =============================================================================
# [ 참고 : 최대, 최소를 갖는 index 출력 함수 정리 ] 
#
# s1 = Series([1,3,10,2,5], index=['a','b','c','d','e'])
# 
# # 1. whichmax, whichmin in R
# # 2. argmax, argmin in python(numpy)
# # 3. idxmax, idxmin in python(pandas)
# 
# s1.argmax()  # 위치값 리턴
# s1.idxmax()  # key값 리턴
# 
# =============================================================================

# 4) 각 일별 지출비용이 가장 높은 두 개 품목 출력                                        # *
s1.sort_values?       # Series 정렬 시 by 옵션 필요 X
card.sort_values?     # DataFrame 정렬 시 by 옵션 필요

card.iloc[0,:].sort_values(ascending=False)[:2].index
card.iloc[1,:].sort_values(ascending=False)[:2].index        # index 리턴
card.iloc[2,:].sort_values(ascending=False)[:2].index.values # 리스트 리턴
Series(card.iloc[2,:].sort_values(ascending=False)[:2].index) # 시리즈 리턴

f4 = lambda x : x.sort_values(ascending=False)[:2].index
f4 = lambda x : x.sort_values(ascending=False)[:2].index.values
f4 = lambda x : Series(x.sort_values(ascending=False)[:2].index)

card.apply(f4, axis=1)                                                         # 202009022

# 9.2 'disease.txt' 파일을 읽고 
df1 = pd.read_csv('disease.txt',sep='\s+', engine='python', encoding = 'euc-kr')
df1.index

# 1) 월별 컬럼 인덱스 설정
df1 = df1.set_index('월별')

# 2) index와 column 이름을 각각 월, 질병으로 저장
df1.index.name = '월'
df1.columns.name = '질병'

# 3) NA를 0으로 수정
df1 = df1.fillna(0)

# 4) 대장균이 가장 많이 발병한 달을 출력
df1.index[df1['대장균'].argmax()]
df1['대장균'].idxmax()

# 5) 각 질병 별 발병횟수의 총합을 출력
df1.apply(sum, axis=0)
df1.sum(axis=0)

# 9.3 employment.csv 파일을 읽고
run profile1
emp2 = pd.read_csv('employment.csv', engine='python', encoding = 'euc-kr')

# 1) 년도와 각 항목(총근로일수, 총근로시간...)을 멀티 컬럼으로 설정
# index 설정
emp2 = emp2.set_index('고용형태')

# 멀티 컬럼 설정
# step1) 현재 컬럼 이름 가공(2007.1 => 2007)
c1 = emp2.columns.map(lambda x : x[:4])    # 1 level value

# step2) 현재 첫번째 행 단위 제거(월급여액 (천원) => 월급여액)
c2 = emp2.iloc[0,:].map(lambda x : x.split(' ')[0])

# step3) c1, c2 멀티 컬럼 전달
emp2.columns = [c1,c2]

# step4)  첫번째 행 제거
emp2 = emp2.drop('고용형태', axis=0)

# 2) 모두 숫자 컬럼으로 저장('-'는 0으로 치환)
# '-'를 '0'으로 치환
emp2.applymap(lambda x : x.replace('-','0'))
emp2 = emp2.replace('-',0)

# ',' 제거
emp2 = emp2.applymap(lambda x : str(x).replace(',',''))

# 숫자 변경
emp2 = emp2.astype('float')
# --------------------------------------------------------------------------- #

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 34.
# multi_index.csv 파일을 읽고 멀티 인덱스를 갖는 데이터프레임 변경
df1 = pd.read_csv('multi_index.csv', encoding = 'euc-kr')

df33 = pd.read_csv('multi_index.csv', engine='python')
df33.iloc[:,0] = df33.iloc[:,0].fillna(method='ffill')
df33 = df33.set_index(['Unnamed: 0','Unnamed: 1'])
df33.index.names = ['지역','지점']
df33.columns =['냉장고','냉장고','TV','TV']
df33.columns = [df33.columns, df33.iloc[0,:]]
df33 = df33.iloc[1:,]
df33.columns.names = ['품목','구분']

# 1) 모든 품목의 price 선택
df33.iloc[:,[0,2]]
df33.xs('price', axis=1, level=1)

# 2) A 지점의 price 선택
df33.iloc[[0,2],0]
df33.xs('A', axis=0, level=1).xs('price', axis=1, level=1)

# 3) seoul의 'B' 지점 선택
df33.xs(['seoul', 'B'], axis = 0, level = [0, 1])
df33.iloc[1,:]
df33.loc[('seoul','B'), :]

# 4) 냉장고의 price 선택
df1.냉장고.xs('price', axis = 1)
df33['냉장고']['price']
df33.iloc[:,0]
df33.loc[:,('냉장고','price')]

# 5) 냉장고의 price, TV의 qty 선택
df33.iloc[:,[0,3]]
df33.loc[:,('냉장고','price')]
df33.loc[:,('TV','qty')]
df33.loc[:,[('냉장고','price'),('TV','qty')]] # ****

# 연습 문제 35. 다음의 데이터 프레임을 멀티 인덱스 설정 후
df1 = pd.read_csv('multi_index_ex1.csv',encoding='cp949')

# 인덱스 설정
df1 = df1.set_index(['지역','지역.1'])
df1.index.names = ['구분','상세']

# 컬럼 설정
c1 = df1.columns.map(lambda x : x[:2])
df1.columns = [c1, df1.iloc[0,:]]
df1 = df1.iloc[1:, :]              # 첫번째 행 제거
df1.columns.names = ['지역','지점']

# 1) 컴퓨터의 서울지역 판매량 출력
df1.loc['컴퓨터','서울']

# 2) 서울지역의 컴퓨터의 각 세부항목별 판매량의 합계 출력
df1.dtypes
df1 = df1.astype('int')
df1.loc['컴퓨터','서울'].sum(1)

# 3) 각 지역의 A지점의 TV 판매량 출력
df1.loc[('가전','TV'),[('서울','A'),('경기','A'),('강원','A')]].sum()
df1.xs('A', axis=1, level=1).xs('TV', axis=0, level=1).iloc[0,:].sum()

# 4) 각 지역의 C지점의 모바일의 각 세부항목별 판매량 평균 출력
df1.loc['모바일', [('서울', 'C'), ('경기', 'C'), ('강원', 'C')]].mean(1)
df1.xs('C', axis=1, level=1).loc['모바일',:].mean(1)

# 5) 서울지역의 A지점의 노트북 판매량 출력
df1.loc[('컴퓨터', '노트북'), ('서울', 'A')]
df1.loc[:,('서울','A')].xs('노트북', level=1)
df1.xs('A', axis=1, level='지점')
# --------------------------------------------------------------------------- # reviewed 20200922

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 36.
# employment.csv 파일을 읽고 멀티인덱스 설정 후(emp2)
emp2

# 1) 각 년도별 정규근로자와 비정규근로자의 월급여액의 차이 계산
s1 = emp2.xs('월급여액', axis=1, level=1).loc['정규근로자',:]
s2 = emp2.xs('월급여액', axis=1, level=1).loc['비정규근로자',:]

s1 - s2

# 2) 각 세부항목의 평균(총근로일수, 총근로시간)
emp2.mean(axis=1, level=1)
# --------------------------------------------------------------------------- #

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 37.
# df1에서 컬럼의 두 레벨을 치환하여 지점이 상위 레벨로 가도록 전달
# A             B            C
# 서울 경기 강원 서울 경기 강원 서울 경기 강원
df1_1 = df1.sort_index(axis=1, level=[1,0], ascending=[False, True])
df1_1.swaplevel(0,1,axis=1)

# 연습 문제 38. 다음의 데이터프레임을 읽고 날짜, 지점, 품목의 3 level index 설정 후
sales = pd.read_csv('sales2.csv', engine='python', encoding = 'euc-kr')

# 인덱스 설정
sales = sales.set_index(['날짜','지점','품목'])

# 1) 출고 컬럼이 높은 순서대로 정렬
sales.sort_values(by='출고', ascending=False)

# 2) 품목 인덱스를 가장 상위 인덱스로 배치
sales.swaplevel(0,2,axis=0)  # 정렬 필요
sales.sort_index(axis=0, level=[2,1]).swaplevel(0,2,axis=0)  # 정렬 필요
# --------------------------------------------------------------------------- #

# -------------------------------- 실 습 문 제 -------------------------------- #
# 실습문제 10.
# 1. multi_index_ex2 파일을 읽고
test1 = pd.read_csv('multi_index_ex2.csv', engine='python', encoding = 'euc-kr')

# 1) 멀티 인덱스, 컬럼 설정
# index 설정
'1_A'[0]
'1_A'[2]

test1.iloc[:, 0].map(lambda x : x[0])    # Error => 문자타입인데 NA가 있어서 NA만 float로 인지됨
c1 = test1.iloc[:, 0].map(lambda x : str(x)[0])
c2 = test1.iloc[:, 0].map(lambda x : str(x)[2])

test1.set_index(c1, c2)
test1.index = [c1, c2]
test1 = test1.iloc[:, 1:]
test1.index.names = ['월', '지점']

# column 설정
test1.columns

'서울'.isalpha()
','.isalpha()
' .1'.isalpha()

v1 = test1.columns.map(lambda x : x if x.isalpha() else NA)
v2 = Series(v1).fillna(method = 'ffill')

test1.columns = [v2, test1.iloc[0, :]]

test1 = test1.iloc[1:, :]

test1.columns.names = ['지역', '요일']

# 2) 결측치 0으로 수정                                                             # *
test1 = test1.fillna(0)

test1.replace('.',0).replace('-',0).replace('?',0)

test1 = test1.replace(['.',',','?','-'],0) # 여러 old 값을 동일한 값으로 치환 가능
'.,?-'.replace(['.',',','?','-'],'0')      # 문자열 replace 메서드는 불가

test1 = test1.astype('int')

# 3) 각 지점별로 요일별 판매량 총합을 출력
test1.sum(axis=0, level=1).sum(axis=1, level=1)

# 4) 각 월별 판매량이 가장 높은 지역이름 출력                                           # *
test1.sum(axis=0, level=0).idxmax(axis=1)

f1 = lambda x : x.idxmax()[0]
test1.sum(axis=0, level=0).apply(f1, axis=1)

test1.sum(axis=0, level=0).sum(axis=1, level=0).idxmax(axis=1) # 정답

# 2. 병원현황.csv 파일을 읽고                                                       # *
test2 = pd.read_csv('병원현황.csv', engine='python', encoding = 'euc-kr', skiprows=1)

# 1) 다음과 같은 데이터프레임으로 만들어라
#                   2013               
#                 1 2 3 4
# 신경과 강남구
#       강동구
#        ....

# 불필요한 데이터 제외
test2 = test2.loc[test2['표시과목'] != '계', :]
test2 = test2.drop(['항목','단위'], axis=1)

# index 생성
test2 = test2.set_index(['시군구명칭','표시과목'])

# column 생성
'2013. 4/4'[:4] # 년도 추출
'2013. 4/4'[6]  # 분기 추출

year = test2.columns.map(lambda x : x[:4])
qt   = test2.columns.map(lambda x : x[6])

test2.columns = [year, qt]
test2.dtypes

# level 치환
test2.sort_index(axis=0, level=1)
test2 = test2.sort_index(axis=0, level=1).swaplevel(0,1, axis=0)

# 2) 성형외과의 각 구별 총 합을 출력
# sol1) 시군구명칭, 표시과목이 index가 아닌 상황(일반 컬럼)
test3 = test2.reset_index()       # index로 설정된 값이 다시 컬럼으로 위치

test3 = test3.loc[test3['표시과목'] == '성형외과', :]
test3 = test3.drop('표시과목', axis=1)
test3.set_index('시군구명칭').sum(1)

# 3) 강남구의 각 표시과목별 총 합 출력
test4 = test2.reset_index()
test4 = test4.loc[test4['시군구명칭'] == '강남구', :]
test4 = test4.drop('시군구명칭', axis=1)
test4.set_index('표시과목').sum(1)

# 4) 년도별 총 병원수의 합 출력
test2.sum(axis=1, level=0)
# --------------------------------------------------------------------------- #

# -------------------------------- 연 습 문 제 -------------------------------- # # *
# 연습문제 39. sales2.csv 파일을 읽고 
sales = pd.read_csv('sales2.csv', engine='python')
sales = sales.set_index(['날짜','지점','품목'])

# 1) 다음과 같은 형태로 만들어라
#                 냉장고          tv             세탁기         에어컨
#                 출고 판매 반품  출고 판매 반품  출고 판매 반품  출고 판매 반품
# 2018-01-01  c1 
sales_1 = sales.unstack().sort_index(axis=1, level=1).swaplevel(0,1,axis=1)

sales.stack().unstack(level=[2,3])

# 2) 위의 데이터 프레임에서 아래와 같은 현황표로 출력(총합)
# 출고  ---
# 판매  ---
# 반품  ---
sales_1.sum(axis=1, level=1).sum(0)
# --------------------------------------------------------------------------- #

# -------------------------------- 연 습 문 제 -------------------------------- # # *
# 연습문제 40. movie_ex1.csv 파일을 읽고
movie = pd.read_csv('movie_ex1.csv', engine='python')

# 1) 지역-시도별, 성별 이용비율의 평균을 정리한 교차테이블 생성
movie2 = movie.set_index(['지역-시도','성별'])['이용_비율(%)']

movie2.unstack()                   # index값의 중복으로 인해 처리 불가
movie2.sum(level=[0,1]).unstack()

movie.pivot_table(index='지역-시도', columns='성별', values='이용_비율(%)',
                  aggfunc='sum')

# 2) 일별- 연령대별 이용비율의 평균을 정리한 교차테이블 생성
movie3 = movie.set_index(['일','연령대'])['이용_비율(%)']
movie3.sum(level=[0,1]).unstack()

movie.pivot_table(index='일', columns='연령대', values='이용_비율(%)',
                  aggfunc='sum')

# 3) 년~ 성별까지를 모두 인덱스로 생성, 10일 이전 데이터 선택
a1 = list(movie.columns.values[:-1])
movie4 = movie.set_index(a1)

movie.loc[movie['일'] < 10, :]
movie.loc[movie4.index.get_level_values(2) < 10]

# [ 참고 - multi column일때 sort_values로 정렬할 컬럼 전달 방법 ]
df1.sort_values(by='a', ascending=False)
df2.sort_values(by=[('col1','A'),('col2','A')], ascending=False)
# --------------------------------------------------------------------------- #

# -------------------------------- 연 습 문 제 -------------------------------- # reviewed 1, 20200923
# 연습문제 41.
# 41.1 movie_ex1.csv 파일을 읽고
movie = pd.read_csv('movie_ex1.csv', engine = 'python', encoding = 'euc-kr')

# 1) 지역-시도별, 성별 이용 비율의 평균을 정리한 교차테이블 생성
movie.pivot_table('이용_비율(%)', '지역-시도', '성별', aggfunc = 'sum' )

# 2) 일별-연령대별 이용 비율의 평균을 정리한 교차테이블 생성
movie.pivot_table('이용_비율(%)', '일', '연령대', aggfunc = 'sum' )

# delivery.csv 파일을 읽고
deli = pd.read_csv('delivery.csv', engine = 'python', encoding = 'euc-kr')


# 41.2 시간대별 배달콜수가 가장 많은 업종 1개 출력
deli.pivot_table(index = '업종', columns = '시간대', values = '통화건수', aggfunc = 'sum')

d1 = deli.pivot_table(index = '시간대', columns = '업종', values = '통화건수', aggfunc = 'sum')
d1.idxmax(axis = 1)
# --------------------------------------------------------------------------- #

# -------------------------------- 실 습 문 제 -------------------------------- #
# 실습문제 11.
# 11.1 delivery.csv 파일을 읽고
deli = pd.read_csv('delivery.csv', engine='python')

# 1) 각 시군구별 업종 비율 출력
#        족발/보쌈 중국음식    치킨
# 강남구   31         45      21.5 ....
deli2 =  deli.pivot_table(index='시군구', columns='업종',values='통화건수',
                          aggfunc='sum')

f1 = lambda x : x / x.sum() * 100
deli2.apply(f1, axis=1)

# 2) 각 업종별 통화건수가 많은 순서대로 시군구의 순위를 출력
# ste1) 첫번째 컬럼에 대해 순위 부여
idx1 = deli2.iloc[:,0].sort_values(ascending=False).index
nrow1 = deli2.shape[0]
Series(np.arange(1,nrow1+1), index = idx1)

# step2) 함수 생성
def f2(x) :
    idx1 = x.sort_values(ascending=False).index
    nrow1 = deli2.shape[0]
    s1 = Series(np.arange(1,nrow1+1), index = idx1)
    return s1

# step3) 각 컬럼별 적용
deli2.apply(f2, axis=0)

# apply에 의해 매 반복마다 리턴 객체가 스칼라이면 최종 리턴은 Series
# apply에 의해 매 반복마다 리턴 객체가 Series이면 최종 리턴은 DataFrame

# [ 참고 : rank에 의한 풀이(rank는 뒤에 정리) ]
deli2.rank(ascending=False, axis=0)

# 3) top(data,n=5) 함수 생성, 업종별 통화건수가 많은 top 5 시간대 출력
# step1) 필요한 교차 테이블 생성
deli3 = deli.pivot_table(index='시간대', columns='업종', values='통화건수',
                         aggfunc='sum')

# step2) 첫번째 컬럼에 대해 수행
deli3.iloc[:,0].sort_values(ascending=False)[:5].index

# step3) 함수 생성 및 적용
top = lambda x, n=5 : x.sort_values(ascending=False)[:n].index
deli3.apply(top, n=3, axis=0)    # 업종이 컬럼일 경우 DataFrame 리턴

deli3.T.apply(top, n=3, axis=1)  # 업종이 인덱스일 경우 Series 리턴

f4 = lambda x, n=5 : Series(x.sort_values(ascending=False)[:n].index)
deli3.T.apply(f4, n=3, axis=1)  # f4 함수의 리턴을 Series로 만든 후 적용
                                # DataFrame 리턴

# 11.2 부동산_매매지수현황.csv파일을 읽고
test2 = pd.read_csv('부동산_매매지수현황.csv', engine='python', skiprows=1)

# step1) multi-column 설정
idx2 = test2.columns.map(lambda x : NA if 'Unnamed' in x else x[:2])
c1 = Series(idx2).fillna(method='ffill')
c2 = test2.iloc[0,:]

test2.columns = [c1,c2]
test2 = test2.iloc[2:,:]

test2.columns.names = ['지역','구분']

# step2) 첫번째 컬럼(날짜) index 설정
test2.set_index(NA)            # NA컬럼이 multi-column 이므로
                               # index가 두 level값을 갖는 tuple로 전달

test2.index = test2.iloc[:,0]  # 첫번째 컬럼 색인 결과는 Series 이므로
                               # index가 1차원 형식으로 전달

test2 = test2.iloc[:,1:]
test2.index.name = '날짜'


# 1) 각 월별 지역별 활발함과 한산함지수의 평균을 각각 출력
# step1) 월별 그룹핑을 위한 년,월,일 3 level의 multi-index 생성
vyear = test2.index.map(lambda x : x[:4])
vmonth = test2.index.map(lambda x : x[5:7])
vday = test2.index.map(lambda x : x[8:])

test2.index = [vyear, vmonth, vday]

# step2) 숫자 컬럼으로 변경
test2 = test2.astype('float')

# step3) 월별 평균
test3 = test2.mean(axis=0, level=1)

# 2) 지역별 활발함지수가 가장 높은 년도 출력                                            # *
test4 = test2.xs('활발함', axis=1, level=1).mean(axis=0, level=0)

test4.idxmax(axis=0)
test4.apply(top, n=3, axis=0)
# --------------------------------------------------------------------------- #

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 42. 연습문제 - 적용함수                                                   # *
# 42.1 emp.csv 파일을 읽고
# 사용자가 입력한 증가율이 적용된 연봉 출력(10입력시 10% 증가)
# 1) map 메서드 사용(불가)
f_sal1 = lambda x, y : round(x * (1 + y/100))
emp['SAL'].map(f_sal1, y=10)  # y인자 전달 불가

# 2) map 함수 사용(키워드 인자 전달 불가, 객체 전달 가능)
list(map(f_sal1, emp['SAL'], y=10))              # 불가
list(map(f_sal1, emp['SAL'], [10]))              # 하나만 출력
list(map(f_sal1, emp['SAL'], np.repeat(10,14)))  # 전체 출력

# 3) apply 사용(키워드 인자 전달 가능)
emp['SAL'].apply(f_sal1, y=10)                       # Series에 전달 가능
                                                     # new feature

f_sal2 = lambda x, y : round(x['SAL'] * (1 + y/100)) # ****
emp.apply(f_sal2, y=15, axis=1)

# 42.2 (SAL + COMM) * 증가율이 적용된 연봉 출력                                    
# 단, 증가율은 10번 부서일 경우 10%, 20번은 11%, 30번은 12%(정수출력)
# 단, COMM이 없는 직원은 100부여
# 1) map 함수
def f_sal3(x,y,z) :   # sal, comm, deptno순 입력
    if z == 10 :
        vrate = 1.1
    elif z == 20 :
        vrate = 1.11
    else :
        vrate = 1.12
    vcomm = Series(y).fillna(100)    
    return round((x + vcomm) * vrate)

f_sal3(800,NA,10)      # Series 리턴
f_sal3(800,NA,10)[0]   # scalar 리턴

list(map(f_sal3, emp.SAL, emp.COMM, emp.DEPTNO))

# -- 함수 수정
def f_sal4(x,y,z) :   # sal, comm, deptno순 입력
    if z == 10 :
        vrate = 1.1
    elif z == 20 :
        vrate = 1.11
    else :
        vrate = 1.12
    vcomm = Series(y).fillna(100)    
    return round((x + vcomm) * vrate)[0]

list(map(f_sal4, emp.SAL, emp.COMM, emp.DEPTNO))
    
# 2) apply
emp['SAL'].apply(f_sal4, emp.COMM, emp.DEPTNO) # 객체 전달 불가

def f_sal5(x) :   # sal, comm, deptno순 입력
    if x['DEPTNO'] == 10 :
        vrate = 1.1
    elif x['DEPTNO'] == 20 :
        vrate = 1.11
    else :
        vrate = 1.12
    vcomm = Series(x['COMM']).fillna(100)    
    return round((x['SAL'] + vcomm) * vrate)[0]

emp.apply(f_sal5, axis=1)
# --------------------------------------------------------------------------- #

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 43. 연습 문제
emp = pd.read_csv('emp.csv')
gogak = pd.read_csv('gogak.csv', engine='python')
gift = pd.read_csv('gift.csv', engine='python')

# 43.1 emp.csv 파일을 읽고 각 직원의 이름, 연봉, 상위관리자의 이름, 연봉 출력
emp2 = pd.merge(emp, emp, left_on='MGR', right_on='EMPNO',
                suffixes=['_직원','_관리자'],
                how='left')

emp2.loc[:,['ENAME_직원','ENAME_관리자','SAL_직원','SAL_관리자']]

# 43.2 gogak.csv 파일과 gift.csv 파일을 읽고
# 각 고객이 받는 상품이름을 고객이름과 함께 출력
gogak
gift.loc[(gift['G_START'] <= 980000) & (980000 <= gift['G_END']),'GNAME']
gift.loc[(gift['G_START'] <= 73000) & (73000 <= gift['G_END']),'GNAME']
gift.loc[(gift['G_START'] <= 320000) & (320000 <= gift['G_END']),'GNAME']

def f_gift(x) :
    vbool = (gift['G_START'] <= x) & (x <= gift['G_END'])
    gname = gift.loc[vbool, 'GNAME']
    return gname

gogak['POINT'].map(f_gift)        # 각 반복마다 Series형식으로 return
f_gift(gogak['POINT'][0]).iloc[0] # 리턴결과에서 원소만 가져오기 위한 색인

def f_gift2(x) :
    vbool = (gift['G_START'] <= x) & (x <= gift['G_END'])
    gname = gift.loc[vbool, 'GNAME']
    return gname.iloc[0]

gogak['POINT'].map(f_gift2)
# --------------------------------------------------------------------------- #

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 44. 연습문제                                                            # *
# 44.1 업종별 콜수가 가장 많은 요일 출력
df1 = pd.read_csv('delivery.csv', engine = 'python', encoding = 'euc-kr')

# step 1) 요일 추출
df1['일자'].strptime('%Y%m%d')    # Series 객체 전달 불가
datetime.strptime(df1['일자'], '%Y%m%d')    # parsing 대상 리스트 전달 불가

d1 = df1['일자'].map(lambda x : datetime.strptime(str(x), '%Y%m%d'))
d1.strftime('%A')    # 벡터 연산 불가

df1['요일'] = d1.map(lambda x : x.strftime('%A'))

# step 2) 교차 테이블 생성
df1_idx = df1.pivot_table(index = '요일', columns = '업종', values = '통화건수',
                aggfunc = 'sum')

# step 3) 업종별 콜수 많은 요일 출력
df1_idx.idxmax(axis = 0)

# 44.2 movie_ex1.csv 파일을 읽고 요일별 이용비율이 가장 높은 연령대 출력
movie = pd.read_csv('movie_ex1.csv', encoding = 'euc-kr')

# step 1) 분리된 년, 월, 일 결합
    # sol 1 => 문자열 결합(+)의 벡터연산 활용
    movie['년'].astype('str') + '/' + movie['월'].astype('str') + '/' + movie['일'].astype('str')
    
    # sol 2 -> 적용함수 활용
    f1 = lambda x, y, z : str(x) + str(y) + str(z)
    date = list(map(f1, movie.년, movie.월, movie.일))    # mapping
    
    f2 = lambda x : str(x.년) + '/' + str(x.월) + '/' + str(x.일)
    movie.apply(f2, axis = 1)    # apply
    
# step 2) 날짜 parsing
    # sol1)
    date = Series(date).map(lambda x : datetime.strptime(x, '%Y%m%d'))
    
    # sol2)
    date.map(lambda x : datetime.strptime(x, '%Y/%m/%d'))
    
# step 3) 날짜 포맷 변경 (요일 추출)    
movie['요일'] = date.map(lambda x : x.strftime('%A'))

# step 4) 교차 테이블 생성
movie_1 = movie.pivot_table(index = '요일', columns = '연령대', values = '이용_비율(%)',
                            aggfunc = 'sum')

# step 5) idx 사용
movie_1.idxmax(axis = 1)

# 기본 문자열 메서드(벡터 연산 X)
# pandas 문자열 메서드(벡터 연산 O)
# --------------------------------------------------------------------------- #  ---------

# -------------------------------- 실 습 문 제 -------------------------------- #
# 실습문제 12.
# 12.1 교습현황.csv 파일을 읽고
test1 = pd.read_csv('교습현황.csv', engine='python', skiprows=1)
test1.columns

# 구 이름 추출
test1['구'] = test1['교습소주소'].map(lambda x : x[6:9])

# 불필요 컬럼 제외
test1 = test1.drop(['교습소주소', '분야구분', '교습계열'], axis=1)

# multi-index 생성(구,교습과정,교습소명)
test1 = test1.set_index(['구','교습과정','교습소명'])

# 년도, 분기, 월 변수 생성
c1 = test1.columns.map(lambda x : x[:4])
c2 = test1.columns.map(lambda x : x[5:].replace(')',''))
c3 = Series(c2).replace(c2[:12],[1,1,1,2,2,2,3,3,3,4,4,4])

# multi-colum 설정(년,분기,월)œ
test1.columns = [c1,c3,c2]

# 천단위 구분기호 제거 후 숫자 컬럼 변경
test1 = test1.applymap(lambda x : x.replace(',','')).astype('int')

# 1) 교습과정별 분기별 교습 금액의 총 합 출력
test1 = test1 / 1000
test1.sum(axis=0, level=1).sum(axis=1, level=1).stack().reset_index()

# 2) 각 구별, 교습과정별 교습금액의 총 합이 가장 높은 교습소명 출력
test1_1 = test1.sum(axis=1).sum(level=[0,1,2]).unstack().fillna(0)
test1_1.idxmax(1)

# 12.2 movie_ex1.csv 파일을 읽고(20200730 in R) 
test2 = pd.read_csv('movie_ex1.csv', engine='python')

# 요일 컬럼 생성
d1 = test2['년'].astype('str') + '/' + test2['월'].astype('str') + '/' + test2['일'].astype('str')
test2['요일'] = d1.map(lambda x : 
                       datetime.strptime(x, '%Y/%m/%d').strftime('%A'))

# 1) 연령대별 성별 이용비율의 평균을 구하여라
test2_1 = test2.pivot_table(index='연령대', columns='성별', values='이용_비율(%)',
                            aggfunc='sum')  

test2_1.stack().reset_index()

# 2) 요일별 이용비율의 평균을 구하여라.
test2.pivot_table(index='요일', values='이용_비율(%)')

# 12.3 delivery.csv 파일을 읽고
test3 = pd.read_csv('delivery.csv', engine='python', encoding = 'euc-kr')

# 1) 일자별 총 통화건수를 구하여라
test3_1 = test3.pivot_table(index='일자', values='통화건수', aggfunc='sum')

# 2) 음식점별 주문수가 많은 시간대를 출력
# 중국음식 12  600
# 보쌈    18   550
#          ...
# ...     24 30
# step1) 교차 테이블 생성
test3_2 = test3.pivot_table(index='시간대', columns='업종', values='통화건수',
                            aggfunc='sum')

# step2) 위 데이터를 조인 가능한 형태(long data)로 변경
test3_2 = test3_2.stack().reset_index()
test3_2 = test3_2.rename({0:'cnt'}, axis=1)

# step3) 업종별 콜수가 많은 시간대 출력
test3_3 = test3_2.idxmax(0).reset_index()
test3_3 = test3_3.rename({0:'시간대'}, axis=1) # 조인 가능한 형태

# step4) 조인
pd.merge(test3_2, test3_3, on=['시간대','업종'])

# 3) 일자별 전일대비 증감률을 구하여라
test3_1

(46081 - 39653) / 39653 * 100          # 16.21

# sol1) index 수정으로 이전 값 가져오기
s1 = list(test3_1.iloc[:-1,0])
s1 = Series(s1, index = test3_1.index[1:])

# 증감률 계산
(test3_1.iloc[:,0] - s1) / s1 * 100

# sol2) 이전 값 가져오는 shift 메서드
s2 = test3_1.iloc[:,0]

s2.shift(periods,    # 몇 번째 이전 값을 가져올지                                    # *
         freq,       # 날짜 오프셋일 경우 날짜 단위 이동 가능
         axis,       # 기본은 이전 행, 컬럼단위로도 이전값 가져올수있음 (axis=1)
         fill_value) # 이전 값이 없을 경우 NA 대신 리턴 값

(s2 - s2.shift(1)) / s2 * 100
# --------------------------------------------------------------------------- #

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 45. professor.csv 파일을 읽고
pro = pd.read_csv('professor.csv', engine='python')

# 1) email-id 출력
vemail = pro.EMAIL.str.split('@').str[0]

# 2) 입사년도 출력
pro.HIREDATE.str[:4]

# 3) ID의 두번째 값이 a인 직원 출력
pro['ID'].str.startswith('a',1)      # 확인 불가(위치값 전달 불가)
pro.loc[pro['ID'].str[1] == 'a', :]

# 4) email_id에 '-' 포함된 직원 출력
pro.loc[vemail.str.contains('-'), :]
pro.loc[vemail.str.find('-') != -1, :]

# 5) 이름을 다음과 같은 형식으로 변경
#    '홍길동' => '홍 길 동'
pro.NAME.str.cat()          # sep='' 가 기본
pro.NAME.str.join()         # sep 옵션 생략 불가
pro.NAME.str.join(sep=' ')  # 각 Series 원소 별 내부 결합 

# 6) PROFNO 컬럼 이름을 PROFID 컬럼으로 변경  
#    (데이터도 4004 => 004004 로 변경)
pro = pro.rename({'PROFNO':'PROFID'}, axis=1)
pro.PROFID.astype('str').str.pad(6,'left','0')

card1 = pd.read_csv('card_history.csv', engine='python') 
card = card1.set_index('NUM')

card.str.replace(',','')   # 벡터화 내장된 문자열 메서드는 DataFrame 적용 불가
# --------------------------------------------------------------------------- #

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 46. student.csv, exam_01.csv 파일을 읽고
std = pd.read_csv('student.csv', engine='python', encoding = 'euc-kr')
exam = pd.read_csv('exam_01.csv', engine='python', encoding = 'euc-kr')

std2 = pd.merge(std, exam).loc[:,['NAME','GRADE','TOTAL']]

# 1) 각 학년별 평균 시험성적
std2.pivot_table(index='GRADE', values='TOTAL')
std2.groupby('GRADE')['TOTAL'].mean()            # Series 출력
std2.groupby('GRADE')[['TOTAL']].mean()          # DataFrame 출력

# 2) 각 학년별, 성별 시험성적의 최대, 최소값                                            # *
std.JUMIN.astype('str').str[6].replace(['1','2'], ['남자','여자'])               
std2['G1'] = std.JUMIN.astype('str').str[6].map({'1':'남자', '2':'여자'})

std2.groupby(['GRADE','G1'])[['TOTAL']].max()              # DataFrame 출력
std2.groupby(['GRADE','G1'])[['TOTAL']].min()              # DataFrame 출력

std2.groupby(['GRADE','G1'])[['TOTAL']].agg(['min','max']) # DataFrame 출력

# [ 참고 - 특정 컬럼 하나 선택 시 차원 축소 방지 ]
std.iloc[:,0]                # Series 리턴
std.iloc[:,0:1]              # DataFrame 리턴(숫자 슬라이스)
std.loc[:,'STUDNO']          # Series 리턴
std.loc[:,'STUDNO':'STUDNO'] # DataFrame 리턴(문자 슬라이스)

std['STUDNO']                # 하나 key 색인 시 Series
std['STUDNO':'STUDNO']       # key indexing에서 slice 색인 불가
std[['STUDNO']]              # key indexing을 사용한 차원 축소 방지***

# 연습문제 47.
# 47.1 sales3.csv 데이터를 불러와서 
sales3 = pd.read_csv('sales3.csv', engine='python', encoding = 'euc-kr')

# 1) 각 날짜별 판매량의 합계를 구하여라.
sales3.groupby('date')['qty'].sum()

# 2) 각 code별 판매량의 합계를 구하여라.
sales3.groupby('code')['qty'].sum()

# 3) product 데이터를 이용하여 각 날짜별, 상품별 매출의 합계를 구하여라
product = pd.read_csv('product.csv')

sales3_1 = pd.merge(sales3, product)
sales3_1['total'] = sales3_1['qty'] * sales3_1['price']
sales3_1.groupby(['date','product'])['total'].sum()

# 47.2 emp 데이터에서 각 연봉의 등급별 연봉의 평균을 출력
# 단, 연봉의 등급은 3000이상 A, 1500 이상 3000미만 B, 1500미만 C
# [0,1500) , [1500,3000), [3000, 10000)
g1 = np.where(emp.SAL >= 3000, 'A',
                               np.where(emp.SAL >= 1500, 'B', 'C'))

pd.cut(emp['SAL'], 
       bins=[0, 1500, 3000, 10000], 
       right=False,
       labels=['C','B','A'])

emp['SAL'].groupby(g1).mean()
# --------------------------------------------------------------------------- #

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 48. 다음의 데이터를 결합하세요 (emp_1과 emp_2, emp_3)








emp_1 = pd.read_csv('emp_1.csv')
emp_2 = pd.read_csv('emp_2.csv')
emp_3 = pd.read_csv('emp_3.csv')

emp_12 = pd.merge(emp_1, emp_2, on='EMPNO')

emp_12.append(emp_3, ignore_index=True)
pd.concat([emp_12, emp_3], axis=0, ignore_index=True)
# --------------------------------------------------------------------------- #

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 49. 다음의 사용자 정의 함수 생성
# 모듈에서 함수 찾기    
# find_func('pd', 'excel')
read_excel

dir(pd)

import pandas

s1 = Series(dir(pandas))
s1[s1.str.contains('read')]

def find_func(module, function) :
    s1 = Series(dir(module))
    return list(s1[s1.str.contains(function)])
    import numpy
find_func(pandas, 'excel')
find_func(numpy, 'excel')

# --------------------------------------------------------------------------- #

# -------------------------------- 실 습 문 제 -------------------------------- # # * => 이해 안됨
# 실습문제 13.
# 13.1 subway2.csv 파일을 읽고 
sub = pd.read_csv('subway2.csv', engine='python', encoding = 'euc-kr', skiprows = 1)

# 역 이름 채우기
sub['전체'] = sub['전체'].fillna(method = 'ffill')

# 전체, 구분 index 생성
sub2 = sub.set_index(['전체', '구분'])

# 컬럼 이름 변경
c1 = sub2.columns.str[:2].astype('int')
sub2.columns = c1

# 1) 각 역별 승하차의 오전/오후별 인원수를 출력
g1 = np.where(c1 < 12, '오전', '오후')    # 24시가 오후에 포함
g2 = pd.cut(c1, bins = [0, 12, 24],    # (0, 12], (12, 24] -> [0, 12), [12, 24)
            right = False,
            labels = ['오전', '오후']).fillna('오전')

sub2.groupby(g1, axis = 1).sum()
sub2.groupby(g2, axis = 1).sum()

# 2) 각 시간대별 승차인원이 가장 큰 5개의 역이름과 승차인원을 함께 출력
sub3 = sub2.xs('승차', level = 1)

# Answer 1 => 교차테이블로부터 상위 5개 역이름 추출
sub3.iloc[:, 0].sort_values(ascending = False)[:5]    # 5시 시간대 확인

f2 = lambda x : x.sort_values(ascending = False)[:5]
sub3.apply(f2, axis = 0)    # NaN 나와서 불편
sub4 = sub3.apply(f2, axis = 0).stack().sort_index(level = [1, 0]).swaplevel(0, 1)    # 깔끔

sub3.apply(f2, axis = 0).stack()    # Error => sort_values 함수는 groupby에 사용 불가

f3 = lambda x : x.sort_values(ascending = False)
sub4.groupby(level = 0).apply(f3)

# Answer 2 => groupby 결과로부터(stack 처리된) 상위 5개 역이름 추출
sub5 = sub3.stack().sort_index(level = [1, 0]).swaplevel(0, 1)

sub5.groupby(level = 0, group_keys = False).apply(f2)

# 13.2 kimchi_test.csv 파일을 읽고                                                # *** 고난도..
kimchi = pd.read_csv('kimchi_test.csv', engine = 'python', encoding = 'euc-kr')

# 1) 각 년도별 제품별 판매량과 판매금액의 평균
kimchi.groupby(['판매년도', '제품'])[['수량', '판매금액']].mean()

# 2) 각 년도별 제품별 판매처별 판매량과 판매금액 평균
kimchi.groupby(['판매년도', '제품', '판매처'])['수량', '판매금액'].mean()

# 3) 각 김치별로 가장 많이 팔리는 월과 해당 월의 판매량을 김치이름과 함께 출력
kimchi2 = kimchi.groupby(['제품', '판매월'])['수량'].sum()    # 시리즈 리턴
kimchi3 = kimchi.groupby(['제품', '판매월'])[['수량']].sum()  # DF 리턴

# Answer 1 => idxmax와 색인을 통한 값 추
kimchi2.groupby(level = 0).idxmax()    # 시리즈 리턴
kimchi3.groupby(level = 0).idxmax()    # DF로 색인했으므로 DF로 나옴 -> 키가 있는 형식으로 색인하면 안됨

kimchi2.loc[('무김치', 3)]    # 튜플로 색인 가능, multi-index의 색인
kimchi2.loc[kimchi2.groupby(level = 0).idxmax()]    # ** => 시리즈로 색인했으므로 시리즈로 나옴
kimchi3.loc[kimchi3.groupby(level = 0).idxmax()]    # Error => 수량 키 해석 불가
kimchi3.loc[kimchi3.groupby(level = 0).idxmax()['수량']]    # 키 제거 후 가능

# Answer 2 => 정렬 후 판매량이 가장 많은 한 행 출력
kimchi2.xs('총각김치', level = 0).sort_values(ascending = False)[:1]

f1 = lambda x : x.sort_values(ascending = False)[:1]
    
kimchi2.groupby(level = 0).apply(f1)    # groupby 함수에 사용자 정의함수 넣으려면 apply 필요, groupby 컬럼 중복
kimchi2.groupby(level = 0, group_keys = False).apply(f1)    # groupby 컬럼 중복 x

# Answer 3 => 교차표의 idxmax와 조인
kimchi4 = kimchi.pivot_table(index = '판매월', columns = '제품', values = '수량', aggfunc = 'sum')

kimchi5 = kimchi4.idxmax(axis = 0).reset_index().rename({0:'월'}, axis = 1)
kimchi2_1 = kimchi2.reset_index()

pd.merge(kimchi2_1, kimchi5, left_on = ['제품', '판매월'], right_on = ['제품', '월'])

# 13.3 delivery.csv 파일을 읽고
deli = pd.read_csv('delivery.csv', engine = 'python', encoding = 'euc-kr', parse_dates = ['일자'])

# 1) 요일별로 각 업종별 통화건수 총 합 확인
deli['요일'] = deli['일자'].map(lambda x : x.strftime('%A'))

deli.groupby(['요일', '업종'])['통화건수'].sum()

# 2) 평일과 주말(금,토,일) 각 그룹별 시군구별 통화건수 총합 출력                            # **
d1 = {'Monday':'평일', 'Tuesday':'평일', 'Wednesday':'평일', 'Thursday':'평일',
      'Friday':'주말', 'Saturday':'주말', 'Sunday':'주말'}
d2 = {['Monday', 'Tuesday', 'Wednesday', 'Thursday']:'평일',
      ['Friday', 'Saturday', 'Sunday']:'주말'}
g3 = deli['요일'].map(d1)
deli.groupby([g3, '시군구'])['통화건수'].sum()
# --------------------------------------------------------------------------- #

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 50.
# taxi_call.csv 데이터를 사용하여
taxi = pd.read_csv('taxi_call.csv', encoding = 'euc-kr')
# 1) 구별 택시콜이 가장 많은 시간대와 콜 수 함께 출력
taxi2 = taxi.groupby(['발신지_시군구', '시간대'])['통화건수'].sum()
g1 = taxi2.groupby(level = 0).idxmax()

taxi2.loc[g1]

# 2) 다음의 시간대별 통화건수의 총합 출력
#    20 ~ 03시 (야간), 03 ~ 08시 (심야), 08 ~ 15시 (오전), 15 ~ 20 (오후)
taxi['시기'] = np.where(taxi['시간대'] > 3, '심야', 
                       np.where(taxi['시간대'] > 8, '오전', 
                                np.where(taxi['시간대'] > 15, '오후', '야간')))

taxi.groupby(['시기'])['통화건수'].sum()
x = 9
def f1(x) :
    if x >= 3 & x < 8:
        return('심야')
    elif x >= 8 & x < 15:
        return('오전')
    elif x >= 15 & x < 20:
        return('오후')
    else :
        return('야간')   
        
taxi['시간대'].apply(f1)[45]        
taxi['시간대'][45]

b1 = [20, 3, 8, 15, 20]
pd.cut(taxi['시간대'], bins = b1,
       right = True, labels = ['야간', '심야', '오전', '오후'])    # Error => bins must increase monotonically

b2 = [0, 3, 8, 15, 20, 24]

# [3, 8)
c1 = pd.cut(taxi['시간대'], bins = b2, include_lowest = True,    # (-0.001, 3], (3, 8]
            labels = ['야간1', '심야', '오전', '오후', '야간2']) 

c1.replace(['야간1', '야간2'], '야간')             # old value만 리스트 전달 => Error (나는 정상 출력 되었음)
c1 = c1.replace(['야간1', '야간2'], ['야간', '야간'])    # new value도 리스트 전달

taxi['통화건수'].groupby(c1).sum()

# 3) 구별 택시콜이 가장 많은 읍면동 상위 3개와 콜수 함께 출력                               # **
taxi3 = taxi.groupby(['발신지_시군구', '발신지_읍면동'])['통화건수'].sum()

f_sort = lambda x : x.sort_values(ascending = False)[:3]

taxi3.groupby(level = 0, group_keys = False).apply(f_sort)
# --------------------------------------------------------------------------- #

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 51. 다음의 문자열에서 문자 + 숫자 형식으로 된 단어만 추출
str1 = '''adljf abd+123 fieij 1111 abc111 Ac+0192 jknkj
          lkjl asdf+0394 jjj'''

r2 = re.compile('[a-z]+\+[0-9]+', flags = re.IGNORECASE)
r2.findall(str1)

# 정규식 표현을 사용한 그룹핑(findall로 각 그룹 추출)
r3 = re.compile('([a-z0-9]+)@[a-z]+.[a-z]{1,3}', flags = re.IGNORECASE)  
# 이메일 아이디를 찾되, () 안에 부분만 가져오기
r4 = re.compile('([a-z0-9]+)@([a-z]+).([a-z]{1,3})', flags = re.IGNORECASE)  

r1.findall(vemail)
r3.findall(vemail)

t1 = Series(r4.findall(vemail)).str[0]
t2 = Series(r4.findall(vemail)).str[1]
t3 = Series(r4.findall(vemail)).str[2]

DataFrame({'t1':t1, 't2':t2, 't3':t3})

# [ 참고 ]
vstr2 = 'abc@naver.com'

r5 = re.compile('.+@.+', flags = re.IGNORECASE)
r5.findall(vstr2)    # 공백을 포함하지 않는 문자열에서는 이메일 주소 추출 잘됨
r5.findall(vemail)   # 공백을 포함하는 경우는 .의 전달은 x
# --------------------------------------------------------------------------- #

# -------------------------------- 실 습 문 제 -------------------------------- #
# 실습문제 14.
# 1. shoppingmall.txt파일을 읽고 쇼핑몰 웹 주소만 출력(총 25개)
# http://.+ 사용 X

# step 1) 파일 불러오기
# 1-1) 하나의 문자열로 만들기
c1 = open('shoppingmall.txt', encoding = 'euc-kr')
test1 = c1.readlines()
c1.close()

# sol 1) for문을 사용한 문자열 결합
vstr = ''
for i in test1 :
    vstr = vstr + i

# sol 2) cat 메서드로 분리된 원소를 하나의 문자열로 결합
vstr2 = Series(test1).str.cat()

# 1-2) Series로 만들기
test2 = pd.read_csv('shoppingmall.txt', engine = 'python', sep = ';', header = None, encoding = 'euc-kr')
test2 = test2.iloc[:, 0]

# step 2) 패턴 생성
import re

p1 = 'http://[a-z0-9./]+'    # . or /가 들어있다면 다 불러오기
pat1 = re.compile(p1, flags = re.IGNORECASE)
pat2 = re.compile('http://.+', flags = re.IGNORECASE)    # not good => 쇼핑몰 주소 뒤에 다른 문자열이 있다면 .+ 사용할 경우 같이 나옴

# step 3) 패턴 추출
pat1.findall(vstr)

pat1.findall(vstr2)

pat2.findall(vstr2)

test2.str.findall(pat1).str[0].dropna()

# 2. ncs학원검색.txt 파일을 읽고 다음과 같은 데이터 프레임 형식으로 출력
# name        addr        tel         start         end    
# 아이티윌  서울 강남구 02-6255-8001  2018-10-12  2019-03-27
# 아이티윌   ( 서울 강남구 ☎ 02-6255-8002 ) 훈련기관정보보기  훈련기간 : 2018-10-12 ~ 2019-03-27  

# step 1) 파일 불러오기
# sol 1)
c2 = open('ncs학원검색.txt', encoding = 'euc-kr')
test3 = c2.readlines()
c2.close()

test3 = Series(test3)

# sol 2)
test3 = pd.read_csv('ncs학원검색.txt', encoding = 'euc-kr', header = None, sep = ';').iloc[:, 0]

# step 2) 패턴 생성
# 아이티윌   ( 서울 강남구 ☎ 02-6255-8002 ) 훈련기관정보보기  훈련기간 : 2018-10-12 ~ 2019-03-27  

p2 = '(.+) \( (.+) ☎ ([0-9-]+) \) .+ : ([0-9-]+) ~ ([0-9-]+)'
# 한칸 이상의 공백을 위해서 .+가 효과적임
pat3 = re.compile(p2)

# step 3) 패턴 추출
c1 = test3.str.findall(pat3).str[0].dropna().str[0].str.strip()
c2 = test3.str.findall(pat3).str[0].dropna().str[1].str.strip()
c3 = test3.str.findall(pat3).str[0].dropna().str[2].str.strip()
c4 = test3.str.findall(pat3).str[0].dropna().str[3].str.strip()
c5 = test3.str.findall(pat3).str[0].dropna().str[4].str.strip()

# step 4) DataFrame 생성
DataFrame({'name':c1, 'addr':c2, 'tel':c3, 'start':c4, 'end':c5})

# --------------------------------------------------------------------------- #

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 54.
# emp.csv 파일을 읽고,
emp = pd.read_csv('emp.csv', encoding = 'euc-kr')
# 1) 연, 월, 일 각각 추출
emp.HIREDATE.map(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M'))
emp['HIREDATE'] = pd.to_datetime(emp.HIREDATE)

emp.HIREDATE.year     # Series의 날짜에서는 year 전달 불가
emp.HIREDATE[0].year  # scalar의 날짜에서는 year 전달 가능

emp.HIREDATE.map(lambda x : x.year)
emp.HIREDATE.map(lambda x : x.month)
emp.HIREDATE.map(lambda x : x.day)

# 2) 급여 검토일의 요일 출력 (단, 급여 검토일은 입사날짜의 100일 후 날짜)
emp.HIREDATE + 100
(emp.HIREDATE + Day(100)).strftime('%A')  # Series 객체 전달 불가
(emp.HIREDATE + Day(100)).map(lambda x : x.strftime('%A'))

# 3) 입사일로부터의 근무일수 출력 
d1 - emp.HIREDATE # timedelta 객체는 X일 X초로 구분해서 출력
                  # days, seconds라는 메서드로 각각 선택하여 출력 가능

(d1 - emp.HIREDATE).days    # Series 객체 전달 불가(벡터 연산 불가)
(d1 - emp.HIREDATE)[0].days # scalar 객체 전달 가능

(d1 - emp.HIREDATE).map(lambda x : x.days)

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 55.
# movie_ex1.csv 파일을 읽고
# 1) 지역별(지역-시도) 요일별 영화 이용비율의 평균을 구하세요.
movie = pd.read_csv('movie_ex1.csv', engine = 'python', encoding = 'euc-kr')

f_datetime = lambda x, y, z : datetime(x, y, z).strftime('%A')
movie['요일'] = list(map(f_datetime, movie['년'], movie['월'], movie['일']))

movie.groupby(['지역-시도', '요일'])['이용_비율(%)'].sum().unstack()

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 56.
# 부동산_매매지수.csv 파일을 읽고
test1 = pd.read_csv('부동산_매매지수.csv', engine = 'python', encoding = 'euc-kr', skiprows = [0, 2])

# NA 제거
test1 = test1.dropna(how = 'all')

# 1) 2008년 4월 7일부터 관찰된 매주, 각 구별 매매지수 데이터로 표현
date3 = pd.date_range('2008/04/07', periods = test1.shape[0], freq = '7D')

test1.index = date3

# 2) 2017년의 작년(2016년) 대비 상승률 상위 10개 구를 상승률과 함께 출력
# Answer 1 => 각 연도 추출 후 연산
ㅕvrate = (test1['2017'].mean(axis = 0) - test1['2016'].mean(axis = 0)) / test1['2016'].mean(axis = 0) * 100
vrate.sort_values(ascending = False)[:10]

# Answer 2 => 전체 연도의 전년도 대비 매매지수 상승률 계산 후 2017년 선택
test2 = test1.resample('Y').mean()
test3 = ((test2 - test2.shift(1)) / test2.shift(1) * 100)['2017'].T
test3.sort_values(by = '2017-12-31', ascending = False)[:10]    # DF이므로 by값 필요
# --------------------------------------------------------------------------- #

# -------------------------------- 실 습 문 제 -------------------------------- #
# 실습문제 14.
# 1. card_history.csv 파일을 읽고
card = pd.read_csv('card_history.csv', encoding = 'euc-kr')
# 1) 2018년 1월 1일부터 매주 일요일에 기록된 자료 가정, 인덱스 생성
d1 = pd.date_range('2018/01/01', periods = card.shape[0], freq = 'W-SUN')
card.index = d1
card = card.drop('NUM', axis = 1)

# 2) 월별 각 항목의 지출 비율 출력
card = card.applymap(lambda x : int(x.replace(',', '')))
card2 = card.resample('M').sum()

f1 = lambda x : round(x / x.sum() * 100, 2)
card2.apply(f1, axis = 1)

# 3) 일별 데이터로 변경하고, 각 일별 지출내용은 하루 평균 지출값으로 나타낸다
# 예) 1월 7일 14000원이면 1월 1일 ~ 1월 7일 각 2000원씩 기록
card.resample('D').asfreq()    # 1월 1일 ~ 1월 6일 출력 x

d2 = pd.date_range('2018/01/01', '2018/07/29')
card3 = card.reindex(d2)
card3.iloc[:8, :]
card3.fillna(method = 'bfill') / 7

# 2. 병원현황.csv 파일을 읽고
test2 = pd.read_csv('병원현황.csv', encoding = 'euc-kr', skiprows = 1)

# 불필요한 컬럼 제거
test2 = test2.drop(['항목', '단위'], axis = 1)

# 계 데이터 제외
test2 = test2.loc[test2['표시과목'] != '계', :]

# index 생성
test2 = test2.set_index(['시군구명칭', '표시과목'])
test2

# multi-columns 생성
c1 = test2.columns.str[:4]
c2 = test2.columns.str[6]

test2.columns = [c1, c2]

# 1) 구별 연도별 각 표시과목(진료과목)의 이전년도 대비 증가율 출력
# (단, 각 데이터는 누적데이터로 가정)
# 4분기 시점의 데이터 추출
test3 = test2.xs('4', axis = 1, level = 1)
test3.shift(-1, axis = 1)    # 2013년 컬럼 NA 리턴 => 데이터 타입이 동일하지 않으면 문제 발생
test3.dtypes                 # shift로 값을 이동해도 원래 컬럼의 데이터타입 유지
                             # 2013년 컬럼은 원래 정수, 2012년 실수 값을 input 시도
test3 = test3.astype('float')
test3 = test3.fillna(0)
test4 = test3.shift(-1, axis = 1)

((test3 - test4) / test4 * 100).fillna(0)

# 2) 구별 연도별 병원이 생성된 수를 기반, 구별 연도별 가장 많이 생긴 표시과목을 각 구별로 5개씩 병원수와 함께 출력
test5 = test3 - test4
test6 = test5.stack().groupby(level = [0, 2, 1]).sum()

f_sort = lambda x : x.sort_values(ascending = False)[:5]
test6.groupby(level = [0, 1], group_keys = False).apply(f_sort)
# --------------------------------------------------------------------------- #

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 57.
# cctv.csv를 불어오고 각 연도별 검거율 증가추이를 각 구별로 비교할 수 있도록 plot 도표 그리기
cctv = pd.read_csv('cctv.csv', encoding = 'euc-kr')

# 검거율 구하기
cctv['검거율'] = cctv['검거'] / cctv['발생'] * 100

# 교차 테이블 생성
cctv2 = cctv.pivot_table(index = '년도', columns = '구', values = '검거율')

plt.plot(cctv2)
cctv2.plot(title = '구별 검거율',
           xticks = cctv2.index,
           rot = 30,
           fontsize = 8,
           ylim = [0, 150],
           style = '--')
# --------------------------------------------------------------------------- #

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 58.
# kimchi_test.csv 파일을 읽고, 
# 각 월별로 김치의 판매량을 비교할 수 있도록 막대그래프로 표현

test3 = pd.read_csv('kimchi_test.csv', engine='python', encoding = 'euc-kr')

test4 = test3.pivot_table(index='판매월', columns='제품',
                          values='수량', aggfunc='sum')

test4.plot(kind='bar',
           ylim=[0,300000],
           rot=0)

plt.legend(title='김치이름', fontsize=7)
plt.ylabel('판매량')
plt.title('월별 김치 판매량 비교')

plt.bar?
# --------------------------------------------------------------------------- #