# -------------------------------- Basic ------------------------------------ #
# [ 데이터 분석 시 필요 모듈 ]
# Numpy, pandas, scikit-learn, … => anaconda 설치
# Anaconda = 기본 파이썬 프로그램 + 데이터 분석 시 필요 모듈

# [ 파이썬 편집 프로그램 ]
# - Jupyter notebook
# - Spyder
# - Pycharm

# iPython
# Import numpy - array
# Import pandas - data frame

# User 사용자 변수 > 시스템 사용자 변수

# spyder 단축키
# - F9 : 라인단위 실행
# - ctrl + 1    : 라인/선택영역 일반 주석/해제
# - ctrl + 4, 5 : 라인/선택영역 구분 주석/해제
# --------------------------------------------------------------------------- #

# 기초 1. print 및 출력 형식
# - %d : 정수 출력 포맷
# - %s : 문자열 출력 포맷
# - %f : 실수 출력 포맷

# 1) 기본 출력
print('abcd')

# 2) % 포맷 전달 방식
print('%d' % 1)
print('%d + %d = %d' % (1,2,1+2))                   # 1+2=3
print('%d 더하기 %d 는 %d 입니다.' % (1,2,1+2))
print('%.2f 더하기 %.2f 는 %.2f 입니다.' % (1,2,1+2))
print('%.2f 더하기 %.1f 는 %.3f 입니다.' % (1,2,1+2))

print('%5d' % 1)    # "    1"
print('%05d' % 1)   # "00001"

# 3) .format 포맷 전달 방식 (위치 전달 가능)
print("{0:d}".format(100))
print("{0:d} + {2:d} = {1:d}".format(100,300,200))     # 100 + 200 = 300

# 4) 특수기호 전달
print('\')     # \ 출력, 에러 발생
print('\\')    # \ 출력, 정상 출력
print('\'')    # ' 출력
print('a\nb\nc')

# [ 참고 1 : 변수 및 상수의 포맷 변경 ]
a1 = '%.2f' % 1
# in oracle to_char(sal,'9999.99')
# --------------------------------------------------------------------------- #

# 기초 2. 연산 및 형 변환 & 기타
# 1) 산술 연산
# - 사칙연산 가능
# - // : 몫
# - % : 나머지

9 // 2    # 4 (몫)
9 % 2     # 1 (나머지)
2 ^ 4     # R에서의 지수 표현식
2 ** 4    # 파이썬에서의 지수 표현식

# 2) 논리연산자
v1 = 100
(v1 > 50) and (v1 < 150)
(v1 > 50) & (v1 < 150)

(v1 > 50) or (v1 < 150)
(v1 > 50) | (v1 < 150)

not(v1 > 50)    # !(v1 > 50) in R

# 3) 형 확인 및 변환
# int()   : 정수 변환
# float() : 실수 변환
# str()   : 문자열 변환

1 + '1'         # 묵시적 형 변환 발생 X, 수행 에러
1 + int('1')    # 형 변환 후 정상 수행

a1 = 1.45
a2 = '2'

type(a1)
type(a2)

# 4) input : 사용자가 입력한 값 문자형으로 가져오기
v1 = int(input('값을 입력하세요 : '))
v1 + 1
v2 = input('값을 입력하세요 : ')
type(v1)
type(v2)

# 5) deep copy (깊은 복사)
# - 객체의 복사가 원본과 전혀다른 공간(메모리)을 차지하는 형태로 복사되는 경우
# - 파이썬의 기본 복사는 얕은 복사인 경우 많음 (같은 메모리 영역 참조)
# - 원본 데이터에 영향을 주지 않음
l1 = [1,2,3,4]
l2 = l1       # 얕은 복사

l3 = l1[:]    # 깊은 복사 => [:] 전체데이터를 선택한다는 개념인듯

l2[0] = 10    # l2 수정
l1[0]         # 10으로 변경되어 있음

l3[1] = 20    # l3 수정
l1[1]         # 20으로 변경되지 않음

# 6) 주소값 확인 함수
id(l1)  # 1491679385032
id(l2)  # 1491679385032
id(l3)  # 1491679964616

# 7) 패킹과 언패킹
a1 = 1,2,3      # 이렇게 괄호 처리 안해주면 알아서 튜플로 처리함, 패킹
a1
v1,v2,v3 = a1   # v1 = a1[0] , v2 = a1[1], v3 = a1[2], 언패킹

# 8) 대칭 합집합 : A-B + B-A       표현:  A^B

# --------------------------------------------------------------------------- #

# 기초 3. 문자열 관련 표현식
# 3.1 문자열 색인(추출) (<-> R)
v1 = 'abcde'
v1[1]
v1[0:3]

# 3.2 다중 라인 문자열 생성
vsql = 'select *
          from emp'        # Error
          
vsql = '''select * 
            from emp'''    # ''' 혹은 """ 안에 넣어야 입력됨  
            
# 3.3 문자열 메서드
s1 = 'abcde'
s2 = 'AbCde'
l1 = ['abc', 'ABC']

# 1) 대소치환
s1.upper()
s1.lower()
s1.title()    # camel 표기법

# 2) startswith : 문자열의 시작 여부 확인
s1.startswith(prefix,  # 시작값 확인문자
              start,   # 검사 시작 위치 (생략 가능)
              end)     # 검사 끝 위치 (생략 가능)

s1.startswith('A')    # False
s1[2] == 'c'
s1.startswith('c', 2)

# 3) endswith : 문자열의 끝 여부 확인
s1.endswith('e')
l1.startswith('a')    # startswith에 l1은 사용 불가 => 적용함수 사용
      
from pandas import Series
s1 = Series([1, 2, 3, 4])

s1.order    # Error => order 함수에 Series 사용 불가
s1.map?     # 함수와 데이터 사용가능여부체크 => map에 s1 사용 가능해? yes!

# 4) strip : 공백 제거
' abcd '.rstrip()    # 오른쪽 공백 제거
' abcd '.lstrip()    # 왼쪽 공백 제거
' abcd '.strip()     # 양쪽 공백 제거

'abcade'.lstrip('a') # 왼쪽에서 'a' 제거
'aaaabcade'.lstrip('a') # 왼쪽에서 'a' 연속으로 제거 (중간은 제거 안됨)

# 5) replace : 치환
'abcde'.replace('a', 'A')    # 치환
'abcde'.replace('c', '')     # 제거

# 6) split : 분리
'a;b;c'.split(';')      # ['a', 'b', 'c']
'a;b;c'.split(';')[1]   # 'b'

# 7) find : 특정문자의 위치값 리턴
'abcde'.find('a')     # 0 => 'a'의 위치 리턴
'abcde'.find('A')     # -1 => 없으면 -1 리턴

'abcde fg 1234'.find('1234')    # 9

# 8) count : 포함 횟수 리턴
'abcabcaa'.count('a')    # 4

# 9) 형(type) 확인
'abd'.isalpha()    # 문자 구성 여부
'abd'.isnumeric()  # 숫자 구성 여부
'abd'.isalnum()    # 문자/숫자 구성 여부

# 10) format : 포맷 변경
'{0:.2f}'.format(10)

# 3.4 문자열 결합
'a' + 'b'

# 3.5 패턴확인 (포함여부) => 정확한 일치여부는 아님
'abcde'.find('c')

'c' in 'abcde'

ename = ['smith', 'allen', 'ward', 'scott']

(ename == 'smith') or (ename == 'allen')    # False

ename in ['smith', 'allen']    # False

# 3.6 문자열 길이 확인
len('abcde')
# --------------------------------------------------------------------------- #
2020-09-15 reviewed 1