# ----------------------------- DATA STRUCTURE ------------------------------ #

# 자료구조 1. 리스트 []
# - 기본 자료 구조
# - R의 벡터와 비슷
# - 1차원
# - 서로 다른 데이터 타입 허용

# 자료구조 2. 튜플 ()

# 자료구조 3. 딕셔너리 {}
# - 기본 자료 구조
# - R의 리스트와 비슷
# - key와 value 형태로 구성
# - key별 서로 다른 데이터 타입 허용

# 자료구조 4. 세트 {}
# - unique

# 자료구조 5. 배열
# - numpy 모듈 지원 자료 구조
# - R의 배열과 동일
# - 다차원
# - 하나의 데이터 타입만 허용

# 자료구조 6. 시리즈
# - pandas 모듈 지원 자료 구조
# - DataFrame을 구성하는 기본 자료 구조 (1차원)
# - 동일한 데이터 타입만 허용
# - key(row number/name) - value 구조
# - index = row number

# 자료구조 7. 데이터프레임
# - pandas 모듈 지원 자료 구조
# - R에서의 데이터프레임과 동일
# - 시리즈의 모음
# - 2차원
# - index(행)와 column(컬럼)으로 구성
# - 행과 열의 구조 => key(column) - value 구조
# --------------------------------------------------------------------------- #

# 자료구조 1. 리스트 []
# - 기본 자료 구조
# - R의 벡터와 비슷
# - 1차원
# - 서로 다른 데이터 타입 허용

# 1.1 생성
l1 = [1, 2, 3]
l2 = [1, 2, 3, '4']
l3 = [1, 2, 3, [4, 5]] ; l3    # 4, 5라는 하나의 리스트가 생성
l3[3]                          # [4, 5]
l3[3][0]                       # 4
                               # 리스트는 중첩구조 허용 O
# c(1, 2, c(3, 4)) => 1 2 3 4  # in R : 벡터는 중첩구조 허용 X

type(l1)    # list
type(l2)    # list

# df1[c(1, 3), ]    # R 방식
df1[[1, 3], ]       # Python 방식

# 1.2 색인
l1[0]
l1[[0, 2]]   # Error => 리스트에서는 리스트로 동시에 여러개 색인 불가
l1[1:2]      # 2번째에서 2번째 원소 추출, 슬라이스 색인은 차원 축소 방지*
l1[1]        # 정수 색인은 하나 추출 시 차원 축소
l1[0:2]      # 1번째에서 2번째 원소 추출
l1[-1]       # reverse indexing (R에서는 제외로 사용)
l1[::2]      # 시작값 : 끝값 + 1 : 증가값 => by 값 전달

# n:m => n ~ (m - 1) 까지 추출

# [ 참고 2 : 시리즈에서의 리스트 색인 ]
# 시리즈 : 1차원이면서 같은 데이터 타입만 허용
from pandas import Series
s1 = Series(l1)
s1[[0, 2]]    # 리스트를 사용한 색인 가능

# 1.3 연산
l1 = [1, 2, 3]
l2 = [10, 20, 30]

l1 + 1    # Error => 리스트와 상수와의 산술연산 불가 (벡터연산 불가)
l1 + l2   # [1, 2, 3, 10, 20, 30] => 리스트 + 리스트는 리스트이 확장
l1 * 3    # l1 + l1 + l1 식으로 확장 => 반복

l1 = [1, 2, 3, 4, 5]
l1 > 3                 # 리스트와 정수의 대소 비교 불가
l1 == 1                # 같다, 같지 않다는 벡터 연산 불가
# 연산을 위해서는 적용함수 사용해야 함

list(map(lambda x : x + 1, l1))

# 1.4 확장
l1 + l2
l1.append(5) ; l1   # 원소를 추가하여 원본 객체를 즉시 수정
l1.append?          # 함수 설명

l2 + [7]        # 추가할 값을 리스트로 전달하여 + 연산 시 추가
l2.extend([7])  # 리스트를 추가하여 원본 객체를 즉시 수정 => 위와 같은 개념으로 []로 묶어줘야 함

# v1 <- c(v1, 4)  # in R

# 1.5 수정
l1[0] = 10 ; l1
l1[1] = [20, 30] ; l1        # 1개에 2개 넣으라고 하면 1개에 리스트 형태로 들어감
l1[2:4] = [300, 400] ; l1    # 개수가 같으면 각각 대입
l1[2:5] = [10, 20] ; l1      # 3개에 2개 넣으라고 하면 2개 다 들어감
l1[2:4] = [10, 20, 30] ; l1  # 2개에 3개 넣으라고 하면 3개 다 들어감

l2[7] = 8    # Error => out of range, 정의되지 않은 position에 값 할당 불가

# 1.6 삭제
del(l1[1])    # 값 삭제 후 원본 수정
del(l1)       # 객체 자체 삭제 가능
l2 = []       # 객체는 유지, 원소만 전체 삭제

# =============================================================================
# # [ 참고 3 : sql, R에서의 문자열 결합 ]
# 'a' || 'b'                  # oracle
# stringr::str_c('a', 'b')    # R
# paste('a', 'b', sep = '')
# =============================================================================

# =============================================================================
# # [ 참고 4 : 함수와 메서드의 차이 ]
# func1(x, y, z) = x + y + z
# func1(data, x, y)
# data.func1(x, y)      # 하나의 인자가 밖으로 빠짐 (주로 데이터)
# =============================================================================

# 1.7 메서드 정리
l1 = [1, 2, 3, 4]

# 1) append => 원소 추가
l1.append(5)

# 2) extend => 리스트 추가
l1 + [6]
l1.extend([6]) ; l1

# 3) insert(위치값, 추가원소) : 특정 위치에 원소 추가 기능
l1.insert(0, 1) ; l1

# 4) remove => 원소값 제거
l2 = ['a', 'b', 'c', 'd']
del(l2[0]) ; l2    # 함수이므로 대상 전체 전달
l2.remove('c') ; l2    # 메서드이므로 대상은 앞에 빠지고 인자에 어떤걸 뺄지 넣으면 됨

# 5) pop(위치값) => 맨 끝부터 원소 제거 (default)
l1.pop() ; l1
l1.pop(0) ; l1    # 0번째 원소 제거

# 6) index(찾는 원소) => 리스트 내 특정 원소의 위치가 리턴
l2.index('b')    # 있으면 위치 리턴
l2.index('e')    # 없으면 에러 처리

# 7) count => 문자용, 숫자용 둘다 있음
l4 = [1, 1, 1, 2, 3]
l4.count(1)

# 8) sort => 값 크기에 따라 정렬되고 수정됨
l5 = [4, 1, 6, 3, 8]
l5.sort() ; l5
l5.sort(reverse = True) ; l5

# 9) reverse => 거꾸로 출력 (원본 수정)
l4[::-1]             # 저장 x
l4.reverse() ; l4    # 저장 o

# 10) len => 문자열 크기 또는 리스트 내 원소의 개수
len(l5)

# [ 참고 8 : 2차원 리스트의 표현 ]
# - 리스트의 중첩 구조를 사용하여 마치 2차원인 것처럼 출력 가능
# - 반복문 필요
l1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
len(l1)    # 3
l1[5, 5]   # Error => 2차원 색인 불가
l1[1][1]   # 순차적 색인 필요

# --------------------------------------------------------------------------- #

# 자료구조 2. 튜플
# - 리스트와 같은 형태이나 수정 불가능한 읽기 전용 객체
# - 원소 삽입, 수정, 삭제 불가
# - 객체 자체 삭제 가능
l1 = [1,2,3,4]
t1 = tuple(l1)
t2 = (1,2,3,4)

type(t1)     # tuple
type(t2)     # tuple

t1.append(5)     #'tuple' object has no attribute 'append'
t1[0] = 10       #'tuple' object does not support item assignment

del(t1[0])       #'tuple' object doesn't support item deletion
del(t1)          # 객체 통째로 삭제는 가능
# --------------------------------------------------------------------------- #

# 자료구조 3. 딕셔너리 {}
# - 기본 자료 구조
# - R의 리스트와 비슷
# - key와 value 형태로 구성
# - key별 서로 다른 데이터 타입 허용
# - 2차원은 아님
# - pandas의 Series, Dataframe의 기본 자료 구조

# 3.1 생성
d1 = {'col1':[1,2,3,4],
      'col2':['a','b']}
type(d1)

# 3.2 색인
d1['col1']           # 색인
d1.get('col1')       # get 메소드 활용

# 3.3 수정
l1[4] = 5            # 추가 불가 Error => list assignment index out of range
d1['col3'] = [1,2]   # key 추가
# 원래 빠른데이터의 삽입과 출력을 위해 키 구조를 만든 것임

d1['col1'][0] = 10   # 이렇게는 수정할 수 잇지롱

from numpy import nan as NA     # nan이 R에서 NA에 해당해서 이런식으로 인식하기 쉽게 이름바꿔 부름

d1['col1'] = NULL    # null이란게 정의되어 있지 않음
d1['col1'] = NA      # NA로 대체, 기 삭제 불가

del(d1['col1']) ; d1

# 3.4 메소드
d1.keys()      # 키값들만 출력
d1.values()    # 값들만 출력
d1.items()     # key와 values를 튜플로 묶어서 출력

# --------------------------------------------------------------------------- #

# 자료구조 4. 세트 {}
# - 딕셔너리의 키 값만 존재
# - 키 값은 중복 될 수 없으므로 중복 불가능한 객체 생성 시 사용
s1 = set(d1)
type(s1)

s2 = {'a','b','c'}  # dictionary에서 데이터값 없이 이렇게 만들면 set로 생성된다.
type(s2)            # set

# 사용되는 이유 : unique같은 느낌과 비슷하게 중복되는걸 알아서 제거하는듯
s3 = {'a','b','c','c'}
s3    # {'a', 'b', 'c'}
# --------------------------------------------------------------------------- #

# 자료구조 5. 배열(array)
# - numpy 모듈 호출 후 사용
# - R의 배열과 동일
# - 다차원 가능
# - 하나의 데이터 타입만 허용

import numpy as np
# 5.1 생성
l1 = [[1,2,3],[4,5,6],[7,8,9]]
l1[0,0]
type(l1)

a1 = np.array(l1)
type(a1)

# 5.2 색인
a1[0,0]
a1[0,:]  # 첫번째 행 선택
a1[:,0]  # 첫번째 컬럼 선택

a1[[0,2], :]   # 리스트 색인 가능(행 선택)

a1[0:2,:]      # 슬라이스 색인 가능

a1[1:3,1:3]        # 행, 열 모두 슬라이스 색인 가능
a1[[1,2],[1,2]]    # 행, 열 모두 리스트 색인 시 point indexing으로 해석
                   # => p(1,1) , p(2,2)

a1[a1[:,1] > 5, :] # boolean indexing 가능
a1[a1 > 5]

# 5.3 배열 메서드
a1.dtype           # 배열을 구성하는 데이터 타입
a1.ndim            # 차원의 수
a1.shape           # 모양

# 5.4 연산
a2 = np.arange(10,100,10).reshape(3,3)
a3 = np.arange(10,130,10).reshape(4,3)

a1 * 9             # 스칼라 연산 가능
a1 + a2            # 서로 같은 크기를 갖는 배열끼리 연산 가능
a1 + a3            # 서로 다른 크기를 갖는 배열끼리 기본적으로 연산 불가

# 5.5 배열의 broadcast 기능 *
# - 서로 다른 크기의 배열 연산 시 크기가 작은 배열의 반복 전달을 가능하게 하는 연산
# - broadcast 전제 조건 필요
# 1) 큰 배열이 작은 배열의 배수 형태
# 2) 작은 배열의 행, 컬럼 중 1의 크기를 가지고 있는 형태

a1 + np.array([10,20,30])   # (3X3) (1X3)
a1 + np.array([10,20,30]).reshape(3,1)   # (3X3) (3X1)

a4 = np.arange(1,17).reshape(4,4)
a5 = np.arange(10,90,10).reshape(2,4)

a4 + a5    # Error => cannot broadcast

# 예제) a1 배열에서 1,3,7,9 추출
a1[c(1,3),c(1,3)]
a1[[0,2],[0,2]]       # 불가

a1[[0,2],:][:,[0,2]]     # 중첩색인으로 가능
a1[np.ix_([0,2],[0,2])]  # np.ix_ 함수로 가능                                     # *

# [ 예제 : 다차원 배열 생성 ]
arr2 = np.arange(1,25).reshape(2,3,4)  # 2(층) X 3(행) X 4(열)
arr2[:,0,:]
arr2[:,0:1,:]
arr2[0,[1,2],2]

# [ 예제 emp.csv 파일을 리스트로 불러온 후 배열로 변경 ]
# emp = np.array(L1)
# ename = emp[1:,1]
# sal = emp[1:,-3]

# 5.6 array 생성 옵션
import numpy as np

# 1) type 지정
l1 = [[1, 2, 3], [4, 5, 6]]
np.array(l1, dtype = float)

# 2) order(배열순서) 지정)
a1 = np.arange(1, 10)

a1.reshape(3, 3, order = 'C')    # 행 우선순위 배치 (기본값))
a1.reshape(3, 3, order = 'F')    # 컬럼 우선순위 배치

# 5.7 형 변환 함수/메서드
# 1) 함수 : int, float, str
# - 벡터연산 불가 -> mapping 처리
float(a1[:, -1])                          # 불가
[float(x) for x in a1[:, -1]]             # 리스트 내포 함수 가능
list(map(lambda x : float(x), a1[:, -1])) # lambda 가능

# 2) 메서드 : astype
# - array, Series, Dataframe에 적용 가능
# - 문자열 혹은 스칼라에는 적용 불가
# - 벡터연산 가능
a1[:, -1].astype('float')

# 5.8 연산
# 1) numpy의 산술연산 함수 및 메서드
a1 = np.array([1, 2, 3])
a2 = np.arange(1, 10).reshape(3, 3)

np.sum(a1)    # 총합
a2.sum()
a2.sum(axis = 0)    # 행별 => 서로 다른 행끼리, 세로방향 (의미적 차이)
a2.sum(axis = 1)    # 열별 => 서로 다른 열끼리, 가로방향

np.mean(a1)   # 평균
a2.mean()

np.var(a1)    # 분산
a2.var()

np.std(a1)    # 표준편차
a2.std()

a2.min(0)
a2.max(1)

a2.cumsum(0)  # 누적합 (세로방향)
a2.cumprod(1) # 누적곱 (가로방향)

a1.argmin()   # 최소값을 갖는 위치값
a1.argmax()   # 최대값을 갖는 위치값

a2.argmin(0)  # 각 컬럼의 최소값 위치 0, 0, 0

# 2) 논리 연산 메서드
(a2 > 5).sum()    # 참의 개수
(a2 > 5).any()    # 하나라도 참인지 여부
(a2 > 5).all()    # 전체가 참인지 여부

# =============================================================================
# # [ 참고 : numpy와 pandas의 분산의 차이 ]
# from pandas import Series

# a1.var()          # 0.67
# s1 = Series(a1)
# s1.var()          # 1

# ((1-2) ** 2 + (2-2) ** 2 + (3-2) ** 2) / 3    # 0.67 => n값 차이
# ((1-2) ** 2 + (2-2) ** 2 + (3-2) ** 2) / 2    # 1    => n값 차이

# sum((a1 - a1.mean()) ** 2) / 3
# sum((a1 - a1.mean()) ** 2) / 2

# vvar = sum(x - xbar) ** 2 / n            # numpy의 분산 계산 식
# vvar = sum((x - xbar) ** 2) / (n - 1)    # pandas의 분산 계산 식 (자유도 n = ddof = 1)
# =============================================================================

# 5.9 array의 deep copy
arr1 = np.array([1, 2, 3, 4])
arr2 = arr1         # 원본 배열의 뷰 생성
arr3 = arr1[0:3]    # 원본 배열의 뷰 생성
arr4 = arr1[:]      # 원본 배열의 뷰 생성   
arr5 = arr1.copy()  # 서로 다른 객체 생성                                          # *

arr1[0] = 10
arr1[1] = 20
arr1[2] = 30
arr2[0]      # arr2도 변경, deep copy 발생 x
arr3[0]      # arr3도 변경, deep copy 발생 x
arr4[1]      # arr4도 변경, deep copy 발생 x
arr4[2]      # arr4도 변경, deep copy 발생 x
arr5[2]      # arr5는 변경되지 x, deep copy 발생 o

# 5.10 np.where
# - R의 ifelse 구문과 비슷
# - np.where(조건, 참리턴, 거짓리턴)
np.where(a1 > 5, 'A', 'B')
np.where(a1 > 5, 'A')    # Error => 거짓리턴 생략 불가

# 5.11 전치 메서드
# 1) T
# - 행, 열 전치 *
# - 3차원 이상은 역 전치

# 2) swapaxes
# - 두 축을 전달 받아 전치
# - 순서 상관 없음

# 3) transpose
# - 여러 축 동시 전치 가능

arr1 = np.arange(1, 10).reshape(3, 3)
arr2 = np.arange(1, 25).reshape(2, 3, 4)

# 2차원 배열의 행, 열 전치
arr1.T
arr1.swapaxes(0, 1)    # 두 축, 순서 상관 없음 
arr1.swapaxes(1, 0) 

arr1.transpose(1, 0)   # 여러 축, 순서 상관 있음
arr1.transpose(0, 1)   # 앞 인자가 행, 뒤 인자가 열

# 3차원 배열의 행, 열 전치
# 3차원에서 T는 행렬 전치가 아니라 역순
arr2.T    # (2층, 3행, 4열) -> (4층, 3행, 2열)

# 층과 열 전치
arr2.swapaxes(0, 2)    # 4층, 3행, 2열

arr2.transpose(층, 행, 열)
arr2.transpose(2, 1, 0)    # 기존의 열값(2), 행은 안바뀌므로 1, 기존의 층값(0)

arr2.transpose(0, 2, 1)    # 행열 전치

# [ 참고 : 축 번호 비교 ]
# in python
# 행 열     층 행 열
# 0 1      0 1  2
# in R
# 행 열     행 열 층
# 1 2      1 2  3

# 5.12 정렬
l1 = [2, 1, 5, 7, 3]
l1.sort() ; l1             # 원본 수정
l1.sort(reverse = True)    # 역순 정렬

arr6 = np.array([2, 3, 6, 9, 1])
arr6.sort() ; arr6         # 처리 가능 => array에 적용 가능한 정렬 메서드
arr6.sort(order = True)    # 오름차순만 가능, 내림차순 전달 불가
arr6.sort(1) ; arr6
a1.sort(0) ; a1            # 정렬방향 지정가능

# 5.13 numpy의 집합 연산자
# 1) union1d : 합집합
# 2) intersect1d : 교집합
# 3) setdiff1d : 차집합
# 4) in1d : 포함 연산자
# 5) setxor1d : 대칭 차집합
# 6) unique : unique value

a1
a22 = a2.reshape(1, -1)    # -1을 쓰면 1번째 인자 1에 맞춰 나머지 세팅

np.union1d(a1, a2)
np.intersect1d(a1, a2)
np.setdiff1d(a2, a1)
np.setxor1d(a1, a2)        # (a1 - a2) U (a2 - a1)

np.in1d(a1, a2)            # 1 in a2, 2 in a2, 3 in a2
[x in a2 for x in a1]     # 위와 같음

np.unique([1, 1, 1, 2])
set([1, 1, 1, 2])
np.unique(np.unique([1, 1, 1, 2]))

# 4) in 연산자 심화
# 1) 문자열 in 연산자 : 문자열 패턴 확인가능 (문자열 안에 해당 문자가 있는지 체크), 벡터 연산 불가
'x' in 'xvg'

# 2) 리스트 in 연산자 : 원소 포함 여부, 벡터 연산 불가
1 in [1, 2]

# 3) np.in1d : array 원소 포함 여부, 벡터 연산 가능
np.in1d([1, 2, 10, 11], [1, 2])

# 4) isin 메서드(in pandas)
s1 = Series([1, 2, 3, 4, 5])
s1.isin([1, 2])    # s1의 매 원소마다 1 혹은 2인지 체크
ename.isin(['SMITH', 'ALLEN'])

# 5.14 NA
# - NA   : 자리수 차지, 결측치 (잘못 입력된 값)
# - NULL : 자리수 차지 x

from numpy import nan as NA
[1, 2, NA]        # 숫자와 NA 함께 표현 가능
['1', '2', NA]    # 문자와 NA 함께 표현 가능

a1.dtype
a1[0] = NA    # Error => cannot convert float NaN to integer
# => NA는 float형임 -> 이미 dtype이 정해져 있다면 NA값 삽입 불가
a2 = a1.astype('float')
a2[0] = NA    # 실수로 변경해주고 시도하면 삽입 가능

# nan 확인 함수
a2[np.isnan(a2)]

np.isnan(a2).sum()
np.isnan(a2).any()

# 5.15 numpy에서의 파일 입출력 함수
np.loadtxt(fname,           # 불러올 파일 이름
           dtype,           # 저장할 데이터 포맷 
           delimiter,       # 구분기호
           skiprows,        # skip할 행 번호(개수)
           usecols,         # 불러올 컬럼
           encoding)        # encoding


np.loadtxt('file1.txt', dtype = 'int', delimiter = ',')
np.loadtxt('file3.txt', dtype = 'int', delimiter = ',', skiprows = 1)
np.loadtxt('file3.txt', dtype = 'int', delimiter = ',', skiprows = 1,
           usecols = [0, 2])

np.savetxt(fname,           # 저장할 파일명 
           X,               # 저장할 객체
           fmt,             # 포맷
           delimiter)       # 분리구분기호

np.savetxt('arr_text1.txt', arr5, delimiter = ';', fmt = '%.2f')
# --------------------------------------------------------------------------- #

# pandas
# - Series, DataFrame 생성 및 연산에 필요한 함수가 내장된 모듈
# - 특히 결측치(NA)에 대한 연산이 빠르고 쉽다
# - 산술연산에 대한 빠른 벡터연산 가능 (함수 및 메서드 제공)
# - 문자열 처리 메서드는 벡터연산 불가 => mapping 필요 (str 메서드 사용 시 가능 **)

# 자료구조 6. 시리즈
# - pandas 모듈 지원 자료 구조
# - DataFrame을 구성하는 기본 자료 구조 (1차원)
# - 동일한 데이터 타입만 허용
# - key(row number/name) - value 구조
# - index = row number

# 6.1 생성
s1 = Series([1, 2, 3, 4])       # dtype = int
s2 = Series([1, 2, 3, 4, '5'])  # dtype = object => 하나의 타입만 가능하므로 문자로 변경
s3 = Series([1, 2, 3, 4, '5'], index = ['a', 'b', 'c', 'd', 'e']) ; s3
s4 = Series([10, 20, 30, 40], index = ['a', 'b', 'c', 'd']) ; s4
s5 = Series([100, 200, 300, 400], index = ['a', 'b', 'c', 'd']) ; s5
s6 = Series([100, 200, 300, 400], index = ['A', 'b', 'c', 'd']) ; s6

# 6.2 연산
s1 + 10    # Series와 스칼라의 산술연산 가능
s4 + s5    # 동일한 크기이면서 index값이 같은 Series끼리 연산 가능
s4 + s6    # 인덱스 매칭되지 않으면 NA 리턴 => 크기가 같지 않아도 연산이 된다는 의미

s1 = Series([1, 2, 3, 4], index = ['a', 'b', 'c', 'd'])
s2 = Series([10, 20, 30], index = ['c', 'a', 'b'])
s1 + s2    # 서로 다른 크기를 갖는 Series라도 동일한 key가 있으면 연산

# 6.3 색인
s1[0]          # 정수색인 (위치색인) 가능
s1[0:1]        # 차원축소 방지를 위한 슬라이스 색인

s1[0:2]        # 슬라이스색인 가능
s1[[0, 1, 2]]  # 리스트 색인 가능
s1[s1 > 3]     # 조건 색인 가능

s1['a']        # 이름 색인 가능
s1.a           # key 색인 (in R : s1$a)

# 6.4 기본 메서드
s1.dtype
s1.dtypes      # 데이터 프레임일때는 복수형만 가능

s1.index       # index(row number / name) 확인
s1.values      # Series를 구성하는 value(데이터) 확인

# 6.5 reindex : index 재배치
s1[['c', 'b', 'a', 'd']]    # c, b, a, d 순 가능
Series(s1, index = ['c', 'b', 'a', 'd'])    # c, b, a, d 순 가능
s1.reindex?
s1.reindex(['c', 'b', 'a', 'd'])    # c, b, a, d 순 가능

# [ 예제 ]
# 다음의 리스트를 금, 화, 수, 월, 목, 일, 토 인덱스값을 갖도록 시리즈로 생성 후 월 ~ 일 순서로 재배치
L1 = [4, 3, 1, 10, 9, 5, 1]
s1 = Series(L1, index = ['금', '화', '수', '월', '목', '일', '토'])
s1.reindex(['월', '화', '수', '목', '금', '토', '일'])
Series(s1, index = ['월', '화', '수', '목', '금', '토', '일'])
# --------------------------------------------------------------------------- #    2020-09-15 reviewed 1

# 자료구조 7. 데이터프레임
# - pandas 모듈 지원 자료 구조
# - R에서의 데이터프레임과 동일
# - 시리즈의 모음
# - 2차원
# - index(행)와 column(컬럼)으로 구성
# - 행과 열의 구조 => key(column) - value 구조

from pandas import DataFrame
# 7.1 생성
d1 = {'col1' : [1, 2, 3, 4], 'col2' : ['a', 'b', 'c', 'd']}
arr1 = np.arange(1, 9).reshape(4, 2)

df1 = DataFrame(d1)
df2 = DataFrame(arr1)
df3 = DataFrame(arr1, index = ['A', 'B', 'C', 'D'], columns = ['col1', 'col2'])

# 7.2 색인
df3[0, 0]    # Error => 기본적 색인 처리 불가
df3['col1']  # 컬럼의 이름 전달 가능

df3.loc      # label indexing (이름 색인)
df3.iloc     # positional indexing (위치 색인)

df3.iloc[0, 0]
df3.iloc[[0, 1], 0]
df3.iloc[[0, 1], 0:2]

df3.iloc[[0, 1], 'col2']    # Error => iloc 메서드에 이름은 전달 불가
df3.loc[[0, 1], 'col2']     # Error => loc 메서드에 위치값은 전달 불가

df3.iloc[df3.col1 > 5, :]   # Error => boolean값 전달은 iloc 메서드로 불가
df3.loc[df3.col1 > 5, :]    # boolean값 전달은 loc 메서드로 가능

df1.iloc[-1, ]    # 제외의 의미가 아님 (reverse 의미)
df1.iloc[1:, :]   # 1번째 행 제외

# drop 메서드
# - 특정 행, 컬럼을 제외
# - 원본 수정 x
# - axis 인자로 행(0), 열(1) 방향 지정
# - 이름으로만 가능 (positional index 값 전달 불가)

df1.drop('c', axis = 0)    # 행 제거 (default)
df1.drop('C', axis = 1)    # 열 제거
df1.drop(0, axis = 1)      # Error => 위치값 전달 불가

df2 = DataFrame(np.arange(1, 5).reshape(2, 2))
df2.drop(0, axis = 0)      # 해당 DF에선 0이 행 이름이므로 가능

# 7.3 기본 메서드
df1.index
df1.columns
df1.values         # key값 제외한 순수 데이터

df1.index.name     # index의 이름
df1.columns.name   # column의 이름

pro.index.name = 'rownum'
pro.columns.name = 'colname'

df1.dtypes

# 7.4 index 수정
df1 = DataFrame(np.arange(1, 17).reshape(4, 4))
df1.index = ['a', 'b', 'c', 'd']
df1.columns = ['A', 'B', 'C', 'D']

# index object의 일부 수정 => 불가
df1.columns[-1] = 'col4'    # Error => index object의 일부 수정불가

# 1) 다른 객체 생성 및 변경 후 index object 덮어쓰기
v1 = df1.columns
v1[-1] = 'col4'    # Error => index object의 일부 수정불가
v2 = df1.columns.values
v2[-1] = 'col4'    # 가능

df1.columns = v2   # 전체 수정 가능

# 2) rename 메서드 활용 (index object 변경 수행)
df1.rename({'C':'col3'})    # 'C'가 index인지 column인지 확인 필요
df1.rename({'C':'col3'}, axis = 0)    # 변경 x, index(행) 이름 수정
df1.rename({'C':'col3'}, axis = 1)    # 변경 o, column(열)) 이름 수정

# 7.5 구조 수정
df1 = DataFrame(np.arange(1, 9).reshape(2, 4))
df2 = DataFrame(np.arange(9, 21).reshape(3, 4))

# 1) row 추가 (rbind)
df1.append([10, 11, 12, 13])    # key error(10-13의 값이 각 컬럼 별로 삽입 x)
df1.append(df2)                 # 행 추가 시 key(컬럼)가 같은 값이 추가
df1.append(df2, ignore_index = True)  # 행 추가 시 기존 index값 무시

# 2) column 추가
df1['4'] = [10, 20]

# 7.6 산술 연산
# - 같은 index, 같은 column끼리 매칭시켜 연산처리
# - 매칭되지 않는 index의 연산결과는 NA
# - 산술연산 메서드(add, sub, mul, div)는 NA로 리턴되는 현상 방지
df1 = df1.drop('4', axis = 1)
df1.columns = ['a', 'b', 'c', 'd']
df2.columns = ['a', 'b', 'c', 'd']

df1 + df2    # 서로 다른 크기의 데이터 프레임 산술연산 가능 (key끼리 매칭)
df2.add(df1, fill_value = 0)    # 차집합은 더 큰 집합으로 채워짐 => 순서 바꿔도 그대로
df2.mul(df1, fill_value = 1)

# 사칙연산 시 fill_value값 전달
# 1 + NA, NA가 0으로 수정
# 1 * NA, NA가 1로 수정
# 1 / NA, NA가 1로 수정
# 1 - NA, NA가 0으로 수정

# [ 참고 : df1 + df2 처리 방식 ]
# 1) df1을 df2처럼 변경 (index가 0, 1, 2가 되도록) => 작은 쪽을 큰 쪽에 맞춰서
df1.reindex(df2.index)

# 2) 위 대상과 df2를 연산처리
df1.reindex(df2.index) + df2

# [ 참고 : numpy와 pandas의 산술연산 메서드 비교 ]
a1 = np.array([4, 1, 10, NA])
a1.mean()                      # numpy의 mean 호출
np.mean(a1)

s1 = Series(a1)
s1.mean()                      # pandas의 mean 호출 (NA 무시가 기본)
s1.mean(skipna = False)        # numpy에서처럼 NA 무시 x

# [ 참고 : in SQL ]
# select avg(comm),           # comm이 있는 직원만 평균 계산
#        sum(comm)/count(*),  # 전체 직원 평균 계산
#        avg(nvl(comm,0))     # 전체 직원 평균 계산
#   from emp;

# 7.7 정렬
emp.sort_values(by,           # 정렬할 컬럼
                axis,         # 정렬 방향(0: 행 정렬, 1: 열 정렬)
                ascending,    # 오름차순 정렬 여부
                inplace,      # 원본 대체 여부                
                na_position)  # NA 배치 순서

emp.sort_values(by = 'ename')
emp.sort_values(by = 'ename', ascending = False)
emp.sort_values(by = ['deptno', 'sal'], ascending = [True, False])
emp.sort_values(by = ['deptno', 'sal'], ascending = [True, False], inplace = True)

# 7.8 reindex 기능
df1 - df2
df1.sub(df2)
df1.sub(df2, fill_value = 0)

df1.reindex?

df1 ** df2
df1 = df1.reindex(df2.index, fill_value = 1)
df1 ** df2

# 7.8 NA 치환
df1 = DataFrame({'col1':[1, NA, 2, NA, 3],
                 'col2':['a','b','c','d',NA]})

# 1) np.where
np.where(pd.isnull(df1.col2), 'e', df1.col2)  # 1차원(컬럼) 가능
np.where(pd.isnull(df1), 'e', df1)            # 2차원 가능

# 2) 조건 치환
df1.col2[pd.isnull(df1.col2)] = 'e'           # 1차원 직접 수정 방식
df1[pd.isnull(df1)] = 'e'                     # 2차원 직접 수정 방식 불가
df1.loc[pd.isnull(df1)] = 'e'                 # 2차원 직접 수정 방식 불가

# 3) 적용함수의 사용
df1.col2.map(lambda x : 'e' if pd.isnull(x) else x) # 1차원 map으로 가능
df1.applymap(lambda x : 'e' if pd.isnull(x) else x) # 2차원 applymap 가능

# 4) NA 치환 함수
df1.col2.fillna('e')  # 1차원(Series) 데이터셋 NA 치환 가능
df1.fillna('e')       # 2차원(DataFrame) 데이터셋 NA 치환 가능

df1.fillna({'col1':0, 'col2':'e'})  # 딕셔너리 전달로 컬럼별 서로 다른 값 치환

df1.fillna(method='ffill')  # 이전 값으로의 치환
df1.fillna(method='bfill')  # 다음 값으로의 치환

# 5) pandas replace 메서드 활용(밑에 정리)
df1.replace(NA,0)

# 7.9 replace 메서드
# 1) 문자열 메서드 형태(기본 파이썬 제공)
# - 문자열 치환만 가능
# - 패턴치환 가능
# - 벡터 연산 불가
# - 문자값 이외 old값 사용 불가
# - 문자값 이외 new값 사용 불가

'abcde'.replace('a','A')
'abcde'.replace('abcde','A')

1.replace(1,0)      # 에러, 숫자에 replace 호출 불가
'1'.replace(1,0)    # 에러, old값은 숫자 불가
'1'.replace('1',0)  # 에러, new값은 숫자 불가
NA.replace(NA,'0')  # 에러, NA값은 치환 불가

df1.applymap(lambda x : x.replace(NA, 0)) # 불가

# 2) pandas 값 치환 메서드 형태(pandas 제공)
# - 값 치환, 패턴치환 불가
# - NA(old value) 치환 가능
# - NA로(new value) 치환 가능
# - 벡터 연산 가능

df1.replace(NA,0)   # pandas에서 제공하는 replace 메서드 호출
df1.replace(1,0)    # 숫자 치환 가능
df1.replace(1,NA)   # NA로의 치환 가능

df1.iloc[0,1] = 'abcde'
df1.replace('a','A')     # 패턴 치환 불가('a'라는 값만 치환 가능) *

# 예제) 
# 아래 데이터 프레임 생성 후
df1 = DataFrame({'a':[10, NA, 9, 1], 
                 'b':['abd','bcd','efc','add']})

# step 1) 10의 값을 100으로 수정
df1[df1 == 10] = 100            # 조건 치환 불가 (R에서는 가능)
df1.loc[df1 == 10] = 100        # 조건 치환 불가 (R에서는 가능)

df1.replace(10, 100)

# step 2) NA값을 0으로 수정
df1.replace(NA, 0)
df1.fillna(0)

# step 3) 데이터프레임에 있는 알파벳 d를 D로 수정
df1.replace('d','D')       # 치환 발생 X (DF에서 패턴 치환은 불가)
'abcd'.replace('d','D')    # 문자열에서는 가능

df1.applymap(lambda x : str(x).replace('d','D'))
df1.b.map(lambda x : str(x).replace('d','D'))

# 7.10 산술연산의 브로드캐스팅 기능 **
# 브로드캐스팅 : 서로 다른 크기의 배열, 데이터프레임이 반복 연산되는 개념
# 1) array에서의 브로드캐스팅 기능
arr1 = np.arange(1,9).reshape(4,2)

arr1 + arr1[:, 0]    # (4X2) + (1X4) => 불가
arr1 + arr1[:, 0:1]  # (4X2) + (4X1) => 가능

# 2) DataFrame에서의 브로드캐스팅 기능
df2 = DataFrame(arr1)

df2 + df2.iloc[:, 0]    # df2의 key(0,1) + Series의 key(0,1,2,3)
df2 + df2.iloc[:, 0:1]  # df2의 key(0,1) + df2.iloc[:,0:1]의 key(0) 

df2 + df2.iloc[0,:]    # df2의 key(0,1) + df2.iloc[0,:]의 key(0,1)

arr1 + arr1[:, 0:1]
df2.add(df2.iloc[:,0], axis = 0)
df2.add(df2.iloc[:,0:1], axis = 0) # add 메서드로 브로드캐스팅 기능 구현 시
                                   # DataFrame + Series 형태여야 함

# =============================================================================
# [ 참고 - 산술연산 메서드 종류 ] 
#
# DataFrame.add(+) : Add DataFrames.
# DataFrame.sub(-) : Subtract DataFrames.
# DataFrame.mul(*) : Multiply DataFrames.
# DataFrame.div(/) : Divide DataFrames (float division).
# DataFrame.truediv : Divide DataFrames (float division).
# DataFrame.floordiv(몫) : Divide DataFrames (integer division).
# DataFrame.mod(나머지) : Calculate modulo (remainder after division).
# DataFrame.pow(지수) : Calculate exponential power.
# 
# =============================================================================