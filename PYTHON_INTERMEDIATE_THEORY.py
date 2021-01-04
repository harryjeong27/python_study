# ----------------------------- Intermediate -------------------------------- #
# 중급 1. 모듈(module)
# - R의 패키지와 같은 개념 (함수의 묶음) (모듈의 묶음 = 패키지)
# - import 명령어로 모듈 호출 후 함수 사용
# - R과는 다르게 내장 모듈(함수)도 호출 후 사용
# - .py 프로그램으로 저장

# 1.1 모듈 호출 방법
# 1) import 모듈명
# 모듈명.함수명

import math
math.trunc(1.98)

# 2) import 모듈명 as 별칭
# 별칭명.함수명
import numpy as np
np.array([1, 2, 3])

# 3) from 모듈명 import 함수명
# 함수명
import pandas as pd
pd.DataFrame({'a1':[1, 2, 3]})

from pandas import DataFrame
DataFrame({'a2':[1, 2, 3]})

from math import *

# math 모듈 내 함수
import math
dir(math)       # 모듈 내 함수 목록 확인
math.sqrt(9)    # 제곱근

2 ** 3          # 거듭제곱 연산자
math.pow(2, 3)  # 거듭제곱 함수

# 1.2 모듈화해서 함수들 보관하기, 사용자 정의 함수의 관리
# 1) 새 페이지에 함수코드를 입력하고, 해당 파일을 일반 파이썬 파일을 저장하듯이 저장,
#    파일명으로 나중에 import 함 (다만, 파이썬의 홈디렉토리에 저장해줘야 함)
# 2) import 파일명        으로 호출        
# 3) 파일명.함수명         으로 해당 함수 사용
# 4) dir(파일명)             으로 해당 모듈에 어떤 함수들이 있는지 확인 가능 
# Tip : 어떤 작업 시에 같이 쓰이는 녀석끼리 잘 묶어서 적당한 이름으로 묶어서 저장하는 것이 좋음
# my_func라는 모듈 만들고 아래 실행
f_add(1,2)

from my_func import *
f_add(1,2)

import my_func

dir(my_func) 

# =============================================================================
# 출력 결과
# '__builtins__',
#  '__cached__',
#  '__doc__',
#  '__file__',
#  '__loader__',
#  '__name__',
#  '__package__',
#  '__spec__',
# ----------------------------------- 이건 기본적으로 있는 것들이고 
# ----------------------------------- 밑에 있는 녀석들이 만든 함수임.
#  'f_add',
#  'f_instr',
#  'f_write_txt'
# =============================================================================

# 1.3 패키지
# - 패키지 > 모듈 > 함수
# - 함수의 묶음이 모듈
# - 모듈의 집합이 패키지
# - 패키지는 하나의 폴더
# - 홈디렉토리에 새폴더를 원하는 패키지명으로 만들고 그 안에 모듈 집어넣으면 됨

# from 패키지명.모듈명 import 함수명
from p1.myfunc 

# 가변형 인자 전달 방식
# - 함수의 인자의 전달 횟수에 제한 두지 않음
# - for문 필요
def f1(x, *iterable) :      # '*'가 가변형 인자임을 나타낸 것이고, 'iterable'은 그 추가적인 값을 담을 통의 이름
    for i in iterable :   
        print(i)

f1(1,2,3,4,5,6)

# 1.4 profile 설정
# - 매 세션마다 호출해야 하는 여러 모듈을 한번에 호출하기 위한 파일
# - .py 파일로 저장

# 1) 새로운 파일 열기
# 2) 호출할 모듈 나열
# 3) 저장
# 4) 새 세션에서 run profile1으로 호출
run profile1
# --------------------------------------------------------------------------- #

# 중급 2. 함수
# 2.1 사용자 정의 함수
# 2.1.1 lambda
# - 축약형 사용자 정의함수 생성 문법
# - 간단한 return만 가능
# - 복잡한 프로그래밍 처리 불가
# - 함수명 = lambda input : output
f1 = lambda x : x + 1     # y = x + 1
f1(5)

f2 = lambda x, y : x + y
f2(4, 5)

f3 = lambda x, y = 0 : x + y    # 뒤에 있는 인자만 default 가능
f3(1)

f3 = lambda x = 0, y : x + y    # Error => 앞에 있는 인자만 default 불가
f3(1)

# map(func, *iterables)    # *가 들어가면 가변형 인자 => ...의 의미 in R
# - 1차원 적용함수 : 데이터 셋의 함수의 반복 적용 가능
# - 결과 출력시 list 함수 사용 필요
# - 함수의 추가 인자 전달 가능
# - 적용 함수 : map => 1차원에서 원소별 반복 적용 도와줌 like sapply
l1 + 1                # 벡터연산 불가
list(map(f1, l1))     # mapping 처리 가능

# 예제) 다음 두 리스트의 각 원소의 합을 출력
l1 = [1, 2, 3, 4, 5]
l2 = [10, 20, 30, 40, 50]

l1 + l2               # 리스트 연산 불가, 확장

f2 = lambda x, y : x + y
list(map(f2, l1, l2))

# 예제) 다음 리스트의 각 원소를 대문자로 변경
l3 = ['a', 'b', 'ab']
l3.upper()    # Error => 리스트에 적용 불가

f3 = lambda x : x.upper()
list(map(f3, l3))

# 예제) 위 문제에서 split 메서드를 사용하여 전달되는 분리구분기호로 분리, 전달되는 위치값에
#      해당되는 원소 추출
f5 = lambda x, y, z : x.split(y)[z]
l1 = ['a;b;c', 'A;B;C']

f5('a;b;c', ';', 1)
f5(l1, ';', 1)             # 불가

list(map(f5, l1, ';', 1))                        # Error

# sol 1 => 인자 값 defaulting
f6 = lambda x, y = ';', z = 0 : x.split(y)[z]
list(map(f6, l1))
# sol 2
# mapping은 각 인자 내 원소 개수가 같아야 함 => 인자 내 각 1번째 원소를 묶어 세트로 전달하기 때문
list(map(f5, l1, [';', ';'], [0, 0]))            

# [ 예제 ]
# 다음의 리스트의 각 원소가 a로 시작하는지 여부 확인
l1 = ['abc', 'ABC', 'bcd']
l1.startswith('a')          # 불가

list(map(lambda x : x.startswith('a'), l1))

l1[[True, False, False]]          # 리스트는 리스트 색인 불가 (조건 색인 불가)
Series(l1)[[True, False, False]]  # 시리즈는 리스트 색인 가능 (조건 색인 가능)

# 2.2 def : 복잡한 프로그래밍 처리 가능한 문법
def f2(x = 0, y = 0) :    # 앞 쪽 인자가 default값 가지면 뒤에 인자도 있어야 함
    return x + y
f2(10, 1)

# 2.3 사용자 정의 함수 유용 표현식
# 2.3.1 for문에 여러 인자 동시 전달 : zip
l1 = [1, 2, 3]
l2 = [10, 20, 30]

for i, j in zip(l1, l2) :    # 중첩문 사용하지 않고도 가능
    print(i + j)
    
# 2.3.2 전역변수 선언 : global
# 1) 전역변수
v1 = 1
def f1() :
    return v1
f1()

# 2) 전역변수보다 지역변수 우선순위 (함수 실행 시 선언하므로)
v1 = 1
def f1() :
    v1 = 10
    return v1
f1()

# 3) 지역변수의 전역변수 선언
def f3() :
    v10 = 10
    return v10

def f4() :
    return v10

f3()    # 10
f4()    # name 'v10' is not defined
v10     # name 'v10' is not defined
    
# 4) 지역변수의 전역변수 설정
def f3() :
    global v10        # v10 <<- 10 (in R)
    v10 = 10    
    return v10

def f4() :
    return v10

f3()    # 10
f4()    # 10
v10     # 10

# 2.4 리스트 내포 표현식(list comprehension)
# - 리스트의 반복 표현식(for)의 줄임 형식
# - lambda + mapping과 비슷           # 비슷한 해결방식 두가지 for문 / list comprehension
map()
[1,2,3,4]

l1 = [1,2,3,4]

# 2.4.1 기본문법 : [리턴대상 for 반복변수 in 반복대상]
[i for i in l1]  # 속도가 빠른편은 아님

# ex
[i * 3 for i in l1]

list(map(lambda i : i * 3, l1))    # 둘이 동일

l2 = []
for i in l1 :
    l2.append(1 * 3)

# 2.4.2 문법 : for 먼저 => [참리턴대상 for 반복변수 in 반복대상 if 조건]
l2 = [1,2,3,4,5]
[i*10 for i in l2 if i > 3]          # [i * 10 /for i in l2 / if i > 3]   의미해석은 이렇게 3등분해서
[i*10 for i in l2 if i > 3 else i * 5] # else return 불가능. 참일 때만 결과 리턴 가능하고 조건이 거짓일 때 경우는 생략됨

list(map(lambda i : i*10 if i > 3 else None, l2)) # else 안쓰면 에러출력됨     # NONE 체크

# 2.4.3 문법 : if 먼저 => [참리턴대상 if 조건 else 거짓리턴대상 for 반복변수 in 반복대상]

[i*10 if i>3 else i*5 for i in l2]   # [i*10 /if i>3 /else i*5 /for i in l2]

list(map(lambda i : i*10 if i > 3 else i*5, l2))  # else 안쓰면 에러출력됨

# --------------------------------------------------------------------------- #

# 중급 3. 반복문 & 조건문
# 3.1 for문
# 3.1.1 기본
# end 어떻게 지정? enter 혹은 depth
# 1) enter
for i in range(0, 10) :
    반복문
    
# 2) depth
for i in range(0, 5) :
    for j in range(0, 5) :
        내부 for문 반복문
    외부 for문 반복문

# =============================================================================
# in R
# for (i in 1:10) {
#         반복문
# }
# =============================================================================

# [ 예제 - 1 ~ 10까지 출력 ]
for i in range(1, 11) :
    print(i)

# enter 없이 print
for i in range(1, 11) :
    print(i, end = '\n')    # default
    
for i in range(1, 11) :
    print(i, end = ' ')     # 1칸의 공백으로 구분

# [ 예제 - 1 ~ 10까지 홀수만 출력 ]
for i in range(1, 11, 2) :
    print(i)

# 3.1.2 중첩 for문
l1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
l2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]]

# 1) 객체 기반        
for i in l1 :
    for j in i :
        print(j, end = ' ')    
    print()
    
for i in l2 :
    for j in i :
        print(j, end = ' ')    
    print()    
    
# flow를 직접 그려보고 하면 쉬움 
# i           j
# [1, 2, 3]   1
#             2
#             3
# (enter)            
# [4, 5, 6]   4       
#             5
#             6

# 2) 위치 기반
for  i in range(0, 3) :
    for j in range(0, 3) :
        print(l1[i][j], end = ' ')
    print()

# i     j
# 0     0   l1[0][0] = 1
#       1   l1[0][1] = 2
#       2   l1[0][2] = 3
# 1     0   l1[1][0] = 4
#       1   l1[1][1] = 5
#       2   l1[1][2] = 6

# 불규칙한 리스트 내 원소 뽑아내기
l2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]]
for i in range(0, len(l2)) :
    for j in range(0, len(l2[i])) :
        print(l2[i][j], end = ' ')
    print()

# 3.2 while문
i = 1
while i < 11 : 
    print(i)
    i = i + 1
    
# 3.3 if문
v1 = 10
if v1 > 3 :
    print('True')
    
if v1 > 3 :
    print('True')
else :
    print('False')

v1 = 15
if v1 > 30 :
    print('30보다 큼')
elif v1 > 10 :
    print('10 ~ 30')
else :
    print('10보다 작음')    

# eval : 전달된 문자열을 명령어로 해석, 처리를 도와주는 함수
v1 = 'print(1)'
eval(v1)

v2 = '(1 + 2 + 10) / 3'
eval(v2)

# 3.4 반복 제어문
# 1) continue : R의 next와 비슷, 특정 반복문만 skip
# 2) break : R의 break, 반복문 자체 종료
# 3) exit : R의 quit과 비슷, 프로그램 자체 종료
# 4) pass

# 1) continue
for i in range(1, 11) :
    if i == 5 :
        continue        # 5일 때 continue를 만나 다음의 구문 skip
    print(i)
print('프로그램 끝')       # 정상 출력

# 2) break
for i in range(1, 11) :
    if i == 5 :
        break           # 5일 때 break를 만나 즉시 반복문 종료
    print(i)
print('프로그램 끝')       # 정상 출력

# 3) exit
for i in range(1, 11) :
    if i == 5 :
        exit(0)         # 5일 때 exit를 만나 즉시 프로그램 종료
    print(i)
print('프로그램 끝')       # 정상 출력 X

# 4) pass
v1 = 'q'
if v1 == 'q' :
    pass
else :
    print('잘못된 입력')

# for :
#     for :
#         if :
#             break
#         print        # break로 인해 실행 x
#     print(values)    # break와 상관없이 계속 반복 수행
# print                # 정상 출력
# --------------------------------------------------------------------------- #

# 중급 4. 파일 입출력
# - open  : 파일을 열어 파일의 내용을 파이썬 메모리영역(커서)에 저장
# - fetch : 커서에 저장된 데이터를 인출(형상화) 
# - close : 커서의 영역 해제, close 하지 않을 경우 메모리누수 현상 발생 가능성
# 4.1 read
# 1)
c1 = open('read_test1.txt')

v1 = c1.readline()    # 한 줄 읽어라
print(v1)

v2 = c1.readline()    # 다음 한 줄 읽어라
print(v2)

c1.close()

# 2)
c1 = open('read_test1.txt')

while 1 :
    v1 = c1.readline()
    if v1 == '' :
        break
    print(v1)

c1.close()

# 3)
c1 = open('read_test1.txt')

while 1 :
    v1 = c1.readline()
    if v1 == '' :
        break
    print(v1, end='')

c1.close()

# 4)
c1 = open('read_test1.txt')
outlist = c1.readlines()    # 리스트로 쭉 불러옴
c1.close()
outlist

# 4.2 write
l1 = [[1,2,3],[4,5,6]]

# 1)
c1 = open('write_test1.txt','w')
c1.writelines(str(l1))
c1.close()        # [[1, 2, 3], [4, 5, 6]]

# 2)
c1 = open('write_test1.txt', 'w')
for i in l1 :
    c1.writelines(str(i) + '\n')
c1.close()        # [[1, 2, 3], [4, 5, 6]]
# --------------------------------------------------------------------------- #
2020-09-15 reviewed 1