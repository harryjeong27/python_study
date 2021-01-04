# ----------------------------- Intermediate -------------------------------- #

# [ 중급 2.1 사용자 정의 함수 & 1차원 적용함수 (lambda & map) ]
# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 4. 다음의 리스트를 생성
L1 = [1, 2, 3, 4]
L2 = [10, 20, 30, 40]
L4 = ['서울', '부산', '대전', '전주']
L5 = ['abc@naver.com', 'a123@hanmail.net']

# 1) L2의 L1승 출력, 10^1, 20^2, 30^3, 40^4    
f1 = lambda x, y : x ** y
list(map(f1, L2, L1))

# 2) L4의 값에 "시"를 붙여 출력
f2 = lambda x : x + '시'
list(map(f2, L4))

# 3) L5에서 이메일 아이디만 출력
# 분리기반
f3 = lambda x : x.split('@')[0]
list(map(f3, L5))

# 위치기반
f4 = lambda x : x[0:x.find('@')]
list(map(f4, L5))
# --------------------------------------------------------------------------- #

# 예제) 위 문제에서 split 메서드를 사용하여 전달되는 분리구분기호로 분리, 전달되는 위치값에
#      해당되는 원소 추출
f5 = lambda x, y, z : x.split(y)[z]
l1 = ['a;b;c', 'A;B;C']

f5('a;b;c', ';', 1)
f5(l1, ';', 1)             # 불가

list(map(f5, l1, ';', 1))  # Error

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
# --------------------------------------------------------------------------- #

# -------------------------------- 실 습 문 제 -------------------------------- #
# 실습문제 1.
# 1) 문자열, 찾을 문자열, 바꿀 문자열을 입력 받아 변경한 결과를 아래와 같이 출력
# 전 : 
# 후 : 
v1 = str(input('문자열 입력 : '))
v2 = str(input('찾을 문자열 입력 : '))
v3 = str(input('바꿀 문자열 입력 : '))    
re1 = v1.replace(v2, v3)

print('전 : %s' % v1)
print('전 : %s' % re1)

# 주의
'%.2f' % 100    # 형태는 실수라도 타입은 문자열

# 2) 이메일 주소를 입력받고 다음과 같이 출력
# 아이디 : a1234
# 메일엔진 : naver

v4 = str(input('이메일 주소 : '))

print('아이디 : ', v4.split('@')[0])
print('메일엔진 : ', v4.split('@')[1].split('.')[0])

# 3) 2번을 활용하여 다음과 같은 홈페이지 주소 출력
# http://kic.com/a1234
print('http://' + v4.split('@')[1] + '/' + v4.split('@')[0])

# 4) num1 = '12,000'의 값을 생성 후, 33으로 나눈 값을 소숫점 둘째자리까지 표현
num1 = '12,000'
num1_1 = float(num1.replace(',', ''))
# Wrong
print('%.2f' %(num1_1 // 33))

# Answer
round(int(num1.replace(',', '')) / 33, 2)

# [ 참고 5 ]
import math
math.trunc(26.987, 2)    # 자리수 전달 불가 (<-> round)

# 5) 다음의 리스트 생성 후 연산
ename = ['smith', 'allen', 'king']
jumin = ['8812111223928', '8905042323343', '90050612343432']
tel = ['02)345-4958', '031)334-0948', '055)394-9050', '063)473-3853']
vid = ['2007(1)', '2007(2)', '2007(3)', '2007(4)']

# 5-1) ename에서 i를 포함하는지 여부 확인
# Answer 1
f1 = lambda x : x.find('i') != -1
list(map(f1, ename))

# [ 참고 6 ]
'i' in 'smith'    # true => 패턴 체크, 있는지 없는지
'i' in ename      # false => 리스트 대상으로는 정확히 'i'라는 원소가 있어?라는 질문
'smith' in ename  # true

# Answer 2
f1 = lambda x : 'i' in x
list(map(f1, ename))

# 5-2) jumin에서 성별 숫자 출력
f2 = lambda x : x[6]
list(map(f2, jumin))

# [ 참고 7 ]
8812111223928[6]    # Error => 숫자형 상태에서는 색인 불가 -> 문자열로 바꿔줘야 함
str(8812111223928)[6]
'8812111223928'[6]

# 5-3) ename에서 smith 또는 allen 인지 여부 출력 [True,True,False]
# Answer 1
f3 = lambda x : x == 'smith' or x == 'allen'
list(map(f3, ename))

# Answer 2
'smith' in ['smith', 'allen']
f3_1 = lambda x : x in ['smith', 'allen']
list(map(f3_1, ename))

# 5-4) tel에서 다음과 같이 국번 XXX 치환 (02)345-4958 => 02)XXX-4958)
# Answer 1 => 위치 기반
f4 = lambda x : x.replace(x[x.find(')') + 1:x.find('-')], 'XXX')
list(map(f4, tel))

# Answer 2 => 분리 기반
f4_1 = lambda x : x.replace(x.split(')')[1].split('-')[0], 'XXX')
list(map(f4_1, tel))

# 5-5) vid 에서 각각 연도와 분기를 따로 저장
# Answer 1 => better
'2007(1)'[:4]
'2007(1)'[5]
f5 = lambda x : x[:4]    # 연도
f5_1 = lambda x : x[5]   # 분기

vyear = list(map(f5, vid))
vqt = list(map(f5_1, vid))

# Answer 2 => part missing
f6 = lambda x : x[0:6].split('(')
list(map(f6, vid))
# --------------------------------------------------------------------------- #

# [ 중급 3. 반복문 ]
# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 5. 다음의 리스트에서 지역번호를 추출 (for문 사용)
tel = ['02)345-9384', '031)3983-3438', '032)348-3938']

for i in range(0, len(tel)) :
    print(tel[i][tel[i].find(')') + 1:tel[i].find('-')])
    
vtel = []
vtel2 = []
for i in tel :
    vno = i.find(')')
    vtel += i[:vno]
    vtel2.append(i[:vno])

# 연습문제 6. 사용자로부터 시작값, 끝값, 증가값을 입력받은 후 시작값부터 끝값 사이의 해당 증가값
#           대상의 총합을 계산 후 출력 (for문 사용))
no1 = int(input('시작값 : '))
no2 = int(input('끝값 : ')) + 1
no3 = int(input('증가값 : '))
    
vsum = 0
for i in range(no1, no2, no3) :
    vsum = vsum + i
    
print('%d에서 %d까지 %d씩 증가값의 총합 : %d' % (no1, no2 - 1, no3, vsum))
  
# 연습문제 7. 원본 문자열과 찾을 문자열, 바꿀 문자열을 차례대로 입력받고 translate 기능으로 각
#           글자마다의 치환 후 결과를 다음과 같이 출력
# (스칼라 테스트 후 리스트 확장)
# 전 : 
# 후 : 
# 'abcdeba', 'abc', '123' => 123de21  
v1.replace(v2[1], v3[1])
v1.replace(v2[1], v3[1]).replace(v2[2], v3[2])
v1.replace(v2[1], v3[1]).replace(v2[2], v3[2]).replace(v2[3], v3[3])

# 원본 리스트, 수정된 리스트 각각 출력
v1 = input('원본 문자열 : ')    # ['abcba', 'abAb'] => 리스트로 들어가지는 않음

l1 = ['abcba', 'abAb']
vold = input('찾을 문자열 : ')
vnew = input('바꿀 문자열 : ')

outlist = []
for i in l1 :
    for j in range(0, len(vold)) :
        i = i.replace(vold[j], vnew[j])
    outlist.append[i]
    
print('전 : %s', % l1)
print('후 : %s', % outlist)

# 연습문제 8. 1부터 100까지의 합
i = 1
vsum = 0
while i < 101 :
    vsum = vsum + i
    i = i + 1

# 연습문제 9.
# 1. 입력한 수식 계산 2. 두 수 사이의 합계 : 1
# *** 수식을 입력하세요 : 3 * 4 / 2 - 5
# 3 * 4 / 2 - 5 결과는 1.0입니다.

# 1. 입력한 수식 계산 2. 두 수 사이의 합계 : 2
# *** 첫번째 숫자를 입력하세요 : 
# *** 두번째 숫자를 입력하세요 :
# 1 + ... + 10는 55입니다.

ans = int(input('1. 입력한 수식 계산\n2. 두 수 사이의 합계 : '))
if ans == 1 :
    str1 = input('계산할 수식을 입력하세요 : ')
    print('%s 결과는  %s입니다.' % (v1_1, eval(str1)))
else :
    no1 = int(input('*** 첫번째 숫자를 입력하세요 : '))
    no2 = int(input('*** 두번째 숫자를 입력하세요 : '))
    vsum = 0
    for i in range(no1, no2 + 1) :
        vsum = vsum + i
    print('%d + ... + %d는 %d입니다.' % (no1, no2, vsum))
# --------------------------------------------------------------------------- #

# -------------------------------- 실 습 문 제 -------------------------------- #
# 실습문제 2.
# 1) 구구단 출력 (중첩 for문)
print('#  2 단  #  #  3 단  #')
for j in range(1, 10):
    for i in range(2, 10) :
        print('%d * %d = %2d' %(i, j, i * j), end = ' ')
    print()

# 2) 별 출력 (while문) *
# unicode 활용
v1 = '\u2605'
print('\u2605')
print('★' * 30)

# Answer 1
star = 14
i = -1
while i < star :
    i = i + 2
    print('-' * int((star - i) / 2) + '★' * i)
    
while i > 1 :
    i = i - 2
    print('-' * int((star - i) / 2) + '★' * i)

# Answer 2
#                             i        공백        별
# print(v1 * 4 + v2 * 1)      1        5 - i      2i - 1
# print(v1 * 3 + v2 * 3)      2
# print(v1 * 2 + v2 * 5)      3
# ...
# print(v1 * 1 + v2 * 7)      6        i - 5      2(10 - i) - 1
# print(v1 * 2 + v2 * 5)      7
# print(v1 * 3 + v2 * 3)      8

v1 = ' '
v2 = '\u2605'

i = 1
while i < 10 :
    if i < 6 :
        print(v1 * (5 - i) + v2 * (2 * i - 1))
    
    else :
        print(v1 * (i - 5) + v2 * (2 * (10 - i) - 1))
    i = i + 1
    
# 3) 사용자로부터 하나의 단어를 입력받고 회문여부 판별
# 회문이란? 앞뒤로 똑같이 생긴 단어를 의미
# ex) allalla
l1 = 'allalla'
len(l1)
l1[0] == l1[-1]
l1[1] == l1[-2]
l1[2] == l1[-3]
l1[3] == l1[-4]

l1 = input('회문 검사용 문자입력 : ')

ans = True
for i in range(0, int(len(l1) / 2) + 1) :
    if l1[i] != l1[- i - 1] :
        ans = False
        break
        
print(ans)

# Answer 2
vstr = input('회문을 판별할 문자열을 입력하세요 : ')
vcnt = len(v3) // 2    # 반복 횟수

vre = 0
for i in range(0, vcnt):
    if vstr[i] == vstr[-(i + 1)] :
       vre = vre + 0    # 앞 뒤 비교가 같으면 0을 더하고
    else :
       vre = vre + 1    # 다르면 1을 누적
if vre == 0 :   
   print('회문입니다.')
else :
   print('회문이 아닙니다.')
# --------------------------------------------------------------------------- # reviewed1 20200907 

# [ 3.4 반복 제어문 ]
# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 10. 1부터 100까지 누적합을 더하다가 3000이 넘는 지점을 출력
#          (해당 지점과 해당지점까지의 누적합 출력)
# i
# 30  2800
# 31  3200
vsum = 0
for i in range(1, 101) :
    vsum = vsum + i
    if vsum > 3000 :
        break
print(i)
print(vsum)

# 연습문제 11. 사용자로부터 값을 입력받아 불규칙한 중첩 리스트를 만드려고 한다.
#            단, 사용자가 종료코드(q)를 입력하면 즉시 종료 후 입력된 불규칙한 리스트를 출력
# l1 = [[1, 2, 3, 4], [5, 6], [7, 8, 9]]
# 1 2 3 4
# 5 6
# 7 8 9        
i = 1
outlist = []
while 1 :
    vstr = input('%d번째 원소를 입력하세요 : ' % i)
    if vstr == 'q' :
        break
    inlist = vstr.split(',')
    outlist.append(inlist)
    i = i + 1
    
for j in outlist :
    for z in j :
        print(z, end = ' ')
    print()
# --------------------------------------------------------------------------- #

# [ 2.2 def : 복잡한 프로그래밍 처리 가능한 문법 ]
# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 12. 다음의 리스트에서 ';'로 분리된 첫번째 값을 추출하는 사용자 정의 함수 생성 및 적용
l1 = ['a;b;c', 'A;B;C']

f1 = lambda x : x.split(';')[0]
list(map(f1, l1))

def f2(x) :
    return x.split(';')[0]
list(map(f2, l1))
# --------------------------------------------------------------------------- #

# [ 중급 4. 파일 입출력 ]
# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 13. 다음의 사용자 정의 함수 생성
#            외부 파일을 불러와 중첩 리스트로 저장하는 함수
# f_read_txt(file, sep = ';', fmt = 'int')
# Answer 1
def f_read_txt(file, sep = ';', fmt = int) :
    c1 = open(file)
    l1 = c1.readlines()
    c1.close()
    
    outlist = []
    
    for i in l1 :
        l2 = i.strip().split(sep)
        inlist=[]
        for j in l2 :
            inlist.append(fmt(j))     
        outlist.append(inlist)   
    
    return outlist

f_read_txt('read_test1.txt', sep=' ', fmt=float) 

# Answer 2) fmt 인자에 문자열 형식으로 전달
def f_read_txt(file, sep=';', fmt='int') :
    c1 = open(file)
    l1 = c1.readlines()
    c1.close()
    
    outlist=[]
    
    for i in l1 :
        l2 = i.strip().split(sep)
        inlist=[]
        for j in l2 :
            vstr = fmt + '(' + str(j) + ')' 
            inlist.append(eval(vstr))     
        outlist.append(inlist)   
    
    return outlist

f_read_txt('read_test1.txt', sep=' ', fmt='int') 

fmt='int'
fmt + '(' + '1' + ')'   #'int(1)'
eval(fmt + '(' + '1' + ')')
# --------------------------------------------------------------------------- #

# -------------------------------- 실 습 문 제 -------------------------------- #
# 실습문제 3.
# 3.1 아래와 같이 변수로 선언된 중첩 리스트를 외부 파일로 저장하는 함수 생성
# f_write_txt(l1, 'write_test1.txt', sep=' ', fmt='%.2f')
l1=[[1,2,3,4],[5,6,7,8]]
'%d' % 1

def f_write_txt(obj, file, sep=' ', fmt='%s') :
    
    c1 = open(file,'w')
    
    for i in obj :
        vstr = ''
        for j in i : 
            vfmt = fmt % j
            vstr = vstr + vfmt + sep
        vstr = vstr.rstrip(sep)
        c1.writelines(vstr + '\n')
        
    c1.close()

f_write_txt(l1, 'write_test3.txt', sep=';', fmt='%.2f')

# 3.2 oracle instr과 같은 함수 생성(없으면 -1 생성)
# f_instr(data,pattern,start=0,      # 0부터 스캔
#                          n=1)      # 첫번째 발견된 pattern의 위치 확인
v1='1#2#3#4#'
v1.find('#')     # 처음(0)부터 스캔해서 첫번째 발견된 # 위치
v1.find('#',2)   # 2 위치부터 스캔해서 첫번째 발견된 # 위치

# 1) n번째 발견된? = 적어도 찾고자 하는 문자열이 n개 포함되어 있다는 의미
v1.count('#') >= n 조건 성립

# 2) n번째 발견된 위치 리턴? instr(v1, '#', 2, 3)
# step1) 첫번째 '#'을 시작위치에서 찾고
v1.find('#',2)  # 3

# step2) 위 위치 다음부터 다시 스캔하여 '#'의 위치 확인
v1.find('#',3+1)  #

def f_instr(data,pattern,start=0,n=1)  :
    vcnt = data.count(pattern)
    if vcnt < n :
        position = -1
    else :
        for i in range(0,n) :
            position = data.find(pattern,start)
            start = position+1
    return position

f_instr('1#2#3#4#','#',start=0,n=1)   # 1
f_instr('1#2#3#4#','#',start=0,n=2)   # 3
f_instr('1#2#3#4#','#',start=0,n=10)  # -1
f_instr('1#2#3#4#','#',start=2,n=1)   # 3
f_instr('1#2#3#4#','#',start=2,n=3)   # 7
# --------------------------------------------------------------------------- #

# 자료구조 3. 딕셔너리 {}
# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 14. 다음의 리스트와 딕셔너리를 참고하여 전화번호를 완성 : 02)345-4958

l1 = ['345-4958', '231-4664', '451-4847', '404-1405']
l2 = ['서울','경기','부산','제주']
area_no = {'서울' : '02', '경기' :'031', '부산':'051', '제주' :'064'}

tel_no = []
for i in range(0,len(l1)):
    tel_no.append(area_no[l2[i]] + ')' + l1[i]) 

tel_no

# -----another 

# zip,       # 검색할 적절한 태그
# 풀이 1
l3 =[]
for i, j in zip(l1,l2) :
    l3.append(area_no.get(j)+ ')' + i)
l3
# 풀이 2
f1 = lambda x, y : area_no.get(y) + ')' + x

list(map(f1,l1,l2))
# --------------------------------------------------------------------------- #

# 자료구조 4. 세트
# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 15. 로또 번호 생성 프로그램
import random
random.randrange(1,46)

lotto = {}
while len(lotto) <= 6 :                          # 6으로 하니깐 7개 나옴
    lotto[random.randrange(1,46)] = []           # lotto.append(random.randrange(1,46)) # str이 아닌 숫자형이여도 문제가 없을지 # append 안됨
                                                    
print('생성된 번호는 ', end='')    
print(lotto, end ='')
print('입니다')

# test
list1 = [1,2,3,4]
s4 = {1,2,3,4}

print ('%d' %list1)  # 안됨

# test
lotto1 = {1,2,3,4}
type(lotto1)
# 숫자형도 상관 없는듯

# another 
lotto = []
while len(lotto) < 6 :
    lotto.append(random.randrange(1,46))

print('추첨된 로또 번호 ===> %s' %lotto)     # 중복발생 가능성 여부

########
lotto = []
while len(lotto) < 6 :
    vno = random.randrange(1,46)
    if vno in lotto :
        pass
    else :
        lotto.append(vno)

lotto.sort()
print("추첨된 로또 번호 ===>%s" %lotto)

# 출력에 리스트의 [] 가 거슬린다면
for i in lotto :
    vstr = vstr + str(i) + ' '

print('추첨된 로또 번호 ===> %s' %vstr)

# set를 활용
lotto = []
while len(lotto) < 6 :
    vno = random.randrange(1,46)
    lotto.append(vno)
    lotto = set(lotto)  # 중복을 제거
    lotto = list(lotto) # set화 되면 append가 안되므로 다시 list로 바꿔주는 과정 필요 , 참고로 set는 sort도 안됨
# --------------------------------------------------------------------------- #

# 중급 2.4 리스트 내포 표현식(list comprehension)
# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 16.                    
sal = ['9,900','25,000','13,000']
addr = ['a;b;c', 'aa;bb;cc', 'aaa;bbb;ccc']
comm = [1000,1600,2000]

# 1) sal의 10% 인상값 출력
[round(int(i.replace(',',''))*1.1,2) for i in sal]

# 2) addr에서 각 두번째값 (b,bb,bbb) 출력
[i.split(';')[1] for i in addr]

list(map(lambda x: x.split(';')[1],addr))

# 3) comm이 1500보다 큰 경우 'A' , 아니면 'B'출력
['A' if i>1500 else 'B' for i in comm]
list(map(lambda i : 'A' if i>1500 else 'B', comm))
# --------------------------------------------------------------------------- #

# 자료구조 3. 딕셔너리 {}
# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 17. 딕셔너리 음식 궁합
foods = {'떡볶이': '튀김',
         '짜장면':'단무지',
         '라면': '김치',
         '피자': '핫소스',
         '맥주': '오징어',
         '치킨': '치킨무',
         '삼겹살' :'버섯'}

while 1 :
    input1 = input('%s 중 좋아하는 음식은?' %foods.keys())
    if input1 == '끝' :
       break
    print('<%s>의 궁합음식은 <%s> 입니다.' %(input1, foods[input1]))

# ------- dict_keys(['떡볶이', '짜장면', '라면', '피자', '맥주', '치킨', '삼겹살']) 중 좋아하는 음식은? 
# 괄호랑 앞에 dict_keys가 붙음    
while 1 :
    flist = list(foods.keys())
    input1 = input('%s 중 좋아하는 음식은?' %flist)
    if input1 == '끝'
        break
    elif input1 in flist :
        print('<%s>의 궁합음식은 <%s> 입니다.' %(input1, foods.get(input1)))
    else :
        print('그런 음식은 없습니다. 확인해보세요')
# --------------------------------------------------------------------------- #

# 2.2 모듈 (함수의 묶음 = 모듈, 모듈의 묶음 = 패키지)
# 2.3 패키지

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 18. 
# 나열된 값의 누적곱을 리턴하는 함수 생성
def f1(*multi) :
    res = 1
    for i in multi:
        res = res* i
    return res

f1(3,4,5)

# 딕셔너리형 인자 전달 방식
f_add(1,2,fmt='%.2f')

def f3(x, **dic) :
    for i in dic.keys() :
        print(dic.get(i))

f3(1, v1= 1, v2= 2, v3 =3)

# 연습문제 19. 
# 두 수를 전달받아 두 수 의 곱을 구하여 리스트에 저장
# 저장된 값은 숫자가 큰 순서대로 정렬하여 출력하도록 하는 사용자 정의함수 생성.
# 단, 사용자 정의함수에 두 수 이외의 reverse라는 키워드 인자를 입력 받도록하자
l_mult = []
def f4(x,y, **dic):
    x*y.append(l_mult)
    if 'reverse' in dic.keys() :
        l_mult.sort(reverse =dic['reverse'])
    else :
        l_mult.sort()

#---------another
fprod(L1, L2, reverse = True)

def fprod(x,y,**dict) :
    vresult = []
    for i,j in zip(x,y) :
       vresult.append(i*j)
    vresult.sort(reverse= dict['reverse']) 
    return vresult

fpod([2,3,4,5],[3,4,5,6], reverse = True)

# os.path.exists(파일명)  파일 존재여부 확인
# --------------------------------------------------------------------------- # 2020-09-16 reviewed

# -------------------------------- 실 습 문 제 -------------------------------- #
# 실습문제 4.
# f_read_txt 사용자 정의함수를 사용하여 emp.csv 파일을 읽고 다음을 수행
from my_func import *
L1 = f_read_txt('emp.csv',sep=',', fmt=str)

L2 = L1[1:]

# 1) 이름을 모두 소문자로 저장
li1 = [[1,2,3],[10,20,30]]
li1[0][0]
li1[1][0]

f1 = lambda x : x[0]
list(map(f1,li1))

# 이름 값 추출
f1 = lambda x : x[1]
ename = list(map(f1,L2))

'A'.lower()

[ x.lower() for x in ename ]

# 2) 입사년도 추출
f1 = lambda x : x[4]
hdate = list(map(f1,L2))

'1981/09/08 00:00:00'[:4]

[x[:4] for x in hdate]

# 3) 10% 인상된 연봉 계산
f1 = lambda x : x[5]
sal = list(map(f1,L2))

int('1100') * 1.1

[int(x) * 1.1 for x in sal]

# 4) comm이 없는 직원은 100 부여
f1 = lambda x : x[-2]
comm = list(map(f1,L2))

[ 100 if x=='' else int(x) for x in comm ]
# --------------------------------------------------------------------------- #

# 5. 배열(array)
# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 20. 
# emp 배열에서
emp = pd.read_csv('emp.csv')
emp1 = np.array(emp.iloc[1:, :])

# 1) sal이 3000이상인 행 선택
emp1[:,-3] >= 3000            # 문자 타입에 대소비교 불가
int(emp1[:,-3])               # int 형 변환함수로 전체 치환 불가
[int(x) for x in emp1[:,-3]]  # list comprehension으로 형 변환 처리

vbool = np.array([int(x) for x in emp1[:,-3]]) >= 3000
emp1[vbool, :]

# 2) comm이 없는 직원의 이름, 부서번호, comm 선택
emp1[emp1[:,-2] == '', :][:,[1,-1,-2]]
emp1[np.ix_(emp1[:,-2] == '',[1,-1,-2])]

# 3) 이름이 S로 시작하는 직원의 이름, 사번, sal 선택
vbool2 = [ x.startswith('S') or x.startswith('s') for x in emp1[:,1]]

emp1[vbool2, :][:,[1,0,-3]]
emp1[np.ix_(vbool2,[1,0,-3])]

# 연습문제 21. 
# 1) 2부터 시작하는 짝수로 구성된 4X4 배열을 만들고
arr1 = np.arange(2,33,2).reshape(4,4)

# 2) 위 배열에 배열의 첫번째 행을 더한 값 출력
arr1 + arr1[0,:]

# 3) 위 배열에 배열의 첫번째 컬럼을 더한 값 출력
arr1 + arr1[:,0:1]
arr1 + arr1[:,0].reshape(4,1)

# 연습문제 22. 
L1   = [[1,2,3],[4,5,6],[7,8,9]]
arr1 = np.array(L1)
arr2 = np.arange(1,25).reshape(2,3,4)

# 1) arr1에서 5,6,8,9 추출
arr1[1:3,1:3]
arr1[1:3,[1,2]]
arr1[[1,2],:][:,[1,2]]
arr1[np.ix_([1,2],[1,2])]

# 2) arr1에서 4,7 추출
arr1[[1,2],0]    # 1차원 리턴
arr1[[1,2],0:1]  # 2차원 리턴

# 3) arr2에서 2,3,6,7,14,15,18,19 출력
arr2[:,0:2,1:3]
arr2[:,[0,1],[1,2]]             # point indexing
arr2[:,[0,1],:][:,:,[1,2]]      # 순차적 색인
arr2[np.ix_(:,[0,1],[1,2])]     # : 사용 불가
arr2[np.ix_([0,1],[0,1],[1,2])] # 정수 리스트 형식으로만 가능

# 4) arr2에서 6,7,8 출력
arr2[0,1,[1,2,3]]
arr2[0,1,1:4]
arr2[0:1,1:2,1:4]

# 연습문제 23. disease.txt 파일을 읽고 (컬럼명 생략) 맨 마지막 컬럼 데이터를 소수점 둘째자리까지 # *
# 표현 후 다시 새로운 파일에 저장
from my_func import *
l1 = f_read_txt('disease.txt', sep = '\t', fmt = str)

l2 = l1[1:]
a1 = np.array(l2)
a1.ndim

# 마지막 컬럼 선택
a1[:, -1]

# 'NA' => '0'
a1[:, -1][-1] = '0'    # 가능하나 반복 작업해야 함
a1[:, -1].replace('NA', '0')    # array에 적용 불가

a2 = [x.replace('NA', '0') for x in a1[:, -1]]

# 소수점 둘째자리 표현
'%.2f' % int(a2[0])

a1[:, -1] = ['%.2f' % int(x.replace('NA', '0')) for x in a1[:, -1]]

f_write_txt(a1, 'disease2.txt')

# 연습문제 24. 다음의 값의 10% 인상된 값 출력
arr2 = np.array(['1,100', '2,200', '3,300'])

[x.replace(',', '').astype('int') for x in arr2] # 불가 => 문자열이라 안됨

arr2_1 = [int(x.replace(',', '')) for x in arr2] # 가능

np.array([x.replace(',', '') for x in arr2]).astype('int') # 가능 => 문자열이라도 배열이면 가능

# 연습문제 25. 다음의 구조를 갖는 array를 생성하자.
# 1   500    5
# 2   200    2
# 3   200    7
# 4    50    9
l1 = [[1, 500, 5], [2, 200, 2], [3, 200, 7], [4, 50, 9]]
a1 = np.array(l1)

# 1) 위의 배열에서 두번째 컬럼 값이 300이상인 행 선택
a1[a1[:, 1] >= 300, :]

# 2) 세번째 컬럼 값이 최대값인 행 선택
a1[a1[:, 2] == a1[:, 2].max(), :]
a1[a1[:, 2].argmax(), :]

# 연습문제 26. 다음의 배열에서 행별, 열별 분산을 구하여라
a1 = np.array([[1, 5, 9], [2, 8, 3], [6, 7, 1]])

# 행별, 열별 평균 구하기
a1.mean(0)
a1.mean(1)

# 행별 연산
a1 - a1.mean(0)                       # 행별 편차
(a1 - a1.mean(0)) ** 2                # 행별 편차 제곱        
np.sum((a1 - a1.mean(0)) ** 2, 0)        # 행별 편차 제곱의 합 
np.sum((a1 - a1.mean(0)) ** 2, 0) / 3    # 행별 분산
a1.var(0)                             # 위와 결과 동일

# 열별 연산
a1 - a1.mean(1)                       # 행별 편차
(a1 - a1.mean(1)) ** 2                # 행별 편차 제곱 => 가로방향으로 누워있음 -> 세로로 변경   
a1 - a1.mean(1).reshape(3, 1)
np.sum((a1 - a1.mean(1)) ** 2, 1)        # 행별 편차 제곱의 합 
np.sum((a1 - a1.mean(1).reshape(3, 1)) ** 2, 1)
np.sum((a1 - a1.mean(1)) ** 2, 1) / 3    # 행별 분산
np.sum((a1 - a1.mean(1).reshape(3, 1)) ** 2, 1) / 3    # 올바른 연산
a1.var(1) 

# 연습문제 27. emp.csv 파일의 부서번호를 사용, 부서이름 출력
#            10이면 인사부 20이면 총무부 30이면 재무부
a_emp = np.array(f_read_txt('emp.csv', sep = ',', fmt = str))

# Answer 1
f1('10')       # 스칼라에 대해 조건 치환 가능
f1(vdeptno)    # Error => if문은 벡터 연산 불가
list(map(f1, vdeptno))

# Answer 2
np.where(a_emp[:, 7] == '10', '인사부',
                             np.where(a_emp[:, 7] == '20', '총무부', '재무부'))

# 연습문제 28. 1 ~ 25의 값을 갖는 5 x 5 배열을 생성 후 2의 배수와 3의 배수를 추출
s1 = np.arange(1, 26).reshape(5, 5)

s2 = s1[s1 % 2 == 0]
s3 = s1[s1 % 3 == 0]

np.union1d(s2, s3)
# --------------------------------------------------------------------------- #

# -------------------------------- 실 습 문 제 -------------------------------- #
# 실습문제 5.
# 5.1 1부터 증가하는 3 x 4 x 5 배열 생성 후
run profile1

# Answer 1
arr1 = np.arange(1, 61).reshape(3, 4, 5)

# 1) 모든 값에 짝수는 * 2를 홀수는 * 3을 연산하여 출력
np.where(arr1 % 2 == 0, arr1 * 2, arr1 * 3)    # 벡터 연산 가능

# - if문 
if arr1 % 2 == 0 :    # 불가
    arr1 * 2
else :
    arr1 * 3
    
def f1(x) :
    for i in x :
        if i % 2 == 0:
            return i * 2
        else :
            return i * 3

# 단점 1)        
f1(3)    # Error => for문에 숫자 스칼라가 들어가면 해석이 안됨 -> 사용자 정의 함수에 for문 잘 사용 안함
for i in 3 :      # Error
    print(i)
for i in [3] :    # 가능
    print(i)    

# 단점 2)
f1([3, 6, 7])    # Wrong => for문에 return은 제대로 수행되지 않음 (한번만 수행)
                 # 여러번 반복될때마다 return이 수행되는 구조 -> 하나만 리턴
                 
# 이렇게 수행해야     
def f1(x) :
    outlist = []
    for i in x :
        if i % 2 == 0:
            outlist.append(i * 2)
        else :
            outlist.append(i * 3)
    return outlist
            
f1([1, 2, 3])           # 가능     
f1(arr1)                # Error => 1개의 층을 통째로(arr1[0]) 보내서 error 발생

for i in arr1 :         # Wrong => 차원 축소 -> 각 층마다 출력, 3번 반복
    print(i)
    
for i in [[1, 2, 3], [4, 5, 6]] :    # Wrong =? 6번 반복 x, 2번 반복
    print(i)    
    
for i in '12345' :    # for문에 문자열 스칼라가 들어가면
    print(i)          # 문자열을 각 글자로 분리 후 반복 수행

# ** for i in 뒤에 나오는 데이터의 len을 계산 후 i에 len 개수대로 넣어준다고 생각하면 이해됨

for i in obj :        # i = obj[0], i = obj[1]
    command
    
3[0]    # 불가
'36'[0] # 가능

# => 적용함수 사용하면 해결됨
# --------------------------------------------------------------------------- #
   
# 2) 각 층의 1번째 3번째 행의 2번째 4번째 컬럼 선택하여 NA로 치환                          # *
arr1[:, [0, 2] [1, 3]]    # Wrong => point indexing
arr1[:, [0, 2], :][:, :, [1, 3]]    # 가능
arr1[np.ix_([0, 1, 2], [0, 2], [1, 3])]         # : 사용 불가 -> 리스트만 전달

arr1[np.ix_([0, 1, 2], [0, 2], [1, 3])] = NA    # Error => float로 변경해야 함
arr1 = arr1.astype('float')
arr1[np.ix_([0, 1, 2], [0, 2], [1, 3])] = NA    # 정상 수행

arr1[:, [0, 2], :][:, :, [1, 3]]     # NA 수정이 안됨 => arr1[:, [0, 2], :]에 대해서 수정하는거임 -> 의미 없음
arr2 = arr1[:, [0, 2], :]            # arr1[:, [0, 2], :]를 따로 만들어서 다시 진행
arr2[:, :, [1, 3]]  = NA

# [ 참고 : 객체 수정 여부 ]
l1 = [[1, 2, 3], [4, 5, 6]]
[ x[0] for x in l1 ] = [ 10, 20 ]  # 원본으로부터 파생된 객체 수정 불가

a1 = np.array(l1)    
a1[:, 0] = [ 10, 20 ]              # 원본의 일부 수정 가능

# [ 참고 : astype 메서드를 사용한 형 변환 유형 ]
# 1) int/float/str의 형 유형 전달 방식
arr1.astype('float')

# 2) type code 전달 방식 / size 지정 가능
arr1.astype('S40')    # string 40 bytes
arr1.astype('U40')    # unicode 40 bytes
arr1.astype('f')      # float
arr1.astype('i')      # int

# [ 참고 : np.ix_ 함수의 인자 전달 방식 ]
# - 리스트만 전달 가능
# - 정수 스칼라 전달 불가, 리스트 안에 삽입 후 전달
# - 슬라이스 색인 형태 전달 불가, 리스트로 전달
arr1[np.ix_( 0, [0, 2], [1, 3])]                # Error
arr1[np.ix_([0], [0, 2], [1, 3])]               # 가능

arr1[np.ix_(0:2, [0, 2], [1, 3])]               # Error
arr1[np.ix_([0, 1], [0, 2], [1, 3])]            # 가능

# 3) 위의 수정된 배열에서 NA의 개수 확인
np.isnan(arr1).sum()

# 4) 층별 누적합 확인
arr1.cumsum(0)

# 5.2 emp.csv 파일을 array 형식으로 불러온 뒤 다음 수행 (컬럼명은 제외)
emp1 = np.loadtxt('emp.csv', dtype = 'str', delimiter = ',', skiprows = 1)

# 1) smith와 allen의 이름, 부서번호, 연봉 출력                                        # *
# Answer 1
(emp1[:, 1] == 'SMITH') | (emp1[:, 1] == 'ALLEN')     # 벡터 연산 가능
(emp1[:, 1] == 'SMITH') or (emp1[:, 1] == 'ALLEN')    # Error => 벡터 연산 불가능

# or을 꼭 쓰고 싶다면
f1 = lambda x : (x == 'SMITH') or (x == 'ALLEN')
list(map(f1, emp1[:, 1]))

# in
'SMITH' in ['SMITH', 'ALLEN']    # 스칼라 가능
emp1[: 1] in ['SMITH', 'ALLEN']  # Error => 벡터 연산 불가능

f2 = lambda x : x in ['SMITH', 'ALLEN']
list(map(f2, emp1[: , 1]))

# np.in1d
vbool = np.in1d(emp1[:, 1], ['SMITH', 'ALLEN'])
emp1[vbool, :][:, [1, -1, -3]]
emp1[np.ix_(vbool, [1, -1, -3])]

np.in1d(['SMITH', 'KING', 'ALLEN'], ['SMITH', 'ALLEN'])

# ename in ('SMITH', 'ALLEN')    # 위와 같음 in SQL
# ename %in% c('SMITH', 'ALLEN') # 위와 같음 in R

# Answer 2
v1 = np.in1d(emp1[:, 1], ['SMITH', 'ALLEN'])
emp1[v1, ][:, [1, 7, 5]]

# 2) deptno가 30번 직원의 comm의 총합
# Answer 1 => Wrong
# way 1) replace + map
v1 = emp1[emp1[:, -1] == '30', -2]
f3 = lambda x : int(x.replace('', '0'))
np.array(list(map(f3, v1))).sum()
sum(list(map(f3, v1)))

# [ 참고 : replace 메서드의 값 치환과 패턴치환 ]
'abc'.replace('a', 'A')     # 패턴 치환
'abc'.replace('abc', 'X')   # 패턴 치환
'abcd'.replace('a', 'X')    # 패턴 치환, 값 치환 불가

# 값 치환 : 바꾸고자 하는 대상과 정확하게 일치하면 치환
if 'abc' == 'abc' :
    print('X')

# way 2) np.where
emp1[:, -2] = np.where(emp1[:, -2] == '', '0', emp1[:, -2])
emp1[emp1[:, -1] == '30', -2].astype('int').sum()

# Answer 2
v2 = emp1[emp1[:, 7] == '30', 6]
v2[v2 == '', ] = '0'

v2 = np.array([int(x) for x in v2])
np.sum(v2)

# 5.3 professor.csv 파일을 array 형식으로 불러온 뒤 다음 수행 (컬럼명은 제외)
prof1 = np.loadtxt('professor.csv', dtype = 'str', delimiter = ',', skiprows = 1, encoding = 'euc-kr')

# 1) email_id 출력
# Answer 1
prof1[:, -2]
vid = list(map(lambda x : x.split('@')[0], prof1[:, -2]))

# Answer 2
id1 = prof1[:, 8]
id1 = np.array([x.split('@')[0] for x in id1])

# 2) 홈페이지가 없는 사람들은 다음과 같이 변경 *
# http://www.kic.com//email_id

# Answer 1
# np.where => Error
np.where(prof1[[:, -1] == '', 'http://www.kic.com//' + vid, emp1[:, -1]])    # Error => 'http://www.'가 반복되지 않음

# mapping
f4 = lambda x, y : 'http://www.kic.com//' + y if x == '' else x
vhpage = list(map(f4, prof1[:, -1], vid))

# 원본 배열의 홈페이지 주소 변경
prof1[:, -1] = vhpage    # 데이터가 짤림
prof1[:, -1] = prof1[:, -1].astype('U40')
prof1[:, -1] = vhpage    # 그래도 똑같음 -> 배열은 모든 타입이 같아야 함

prof1.dtype              # 원본을 수정해야 함
prof1 = prof1.astype('U40')
prof1[:, -1] = vhpage    # 정상 출력

# Answer 2
prof1[prof1[:, 9] == '', 9] = np.array(['http://www.' + x for x in id1[prof1[:, 9] == '', ]])

# 3) pay가 최대인 사람의 이름, pay, position 출력
a1.argmax()   # 최대값을 갖는 위치값

prof1[prof1[:, 4].argmax(), ][[1, 4, 3]]
# --------------------------------------------------------------------------- #

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 30.
df1 = pd.read_csv('professor.csv', encoding = 'euc-kr')
pro = pd.read_csv('professor.csv', engine = 'python', encoding = 'euc-kr')

# 1) 홈페이지 주소가 있는 경우 그대로, 없으면                                           # *
#    http://www.kic.com//email_id
# Answer 1
f1 = lambda x, y : 'http://www.kic.com//' + y.split('@')[0] if pd.isnull(x) else x
list(map(f1, pro.HPAGE, pro.EMAIL))

# Answer 2
df1.EMAIL.split('@')
np.where(pd.isnull(df1.HPAGE), 'http://www.kic.com//' + df1.EMAIL.map(lambda x : x.split('@')[0]), df1.HPAGE)

# 2) avg_sal 컬럼에 각 행마다 각 행의 부서번호를 확인 후,                                # *
#    같은 부서의 평균 avg 값을 삽입
pro.loc[pro.DEPTNO == 101, 'PAY'].mean()
pro.loc[pro.DEPTNO == 102, 'PAY'].mean()

f2 = lambda x : pro.loc[pro.DEPTNO == x, 'PAY'].mean()
pro['avg_sal'] = pro.DEPTNO.map(f2)

# 3) index, columns name 설정
pro.index.name = 'rownum'
pro.columns.name = 'colname'
# --------------------------------------------------------------------------- #

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 31.
# 1. emp.csv 파일을 읽고
emp = pd.read_csv('emp.csv')

# 1) index 값을 사원번호로 설정
emp.index = emp.EMPNO
emp = emp.drop('EMPNO', axis = 1)

# 2) 컬럼이름을 모두 소문자로 변경
# Answer 1
emp.columns = emp.columns.map(lambda x : x.lower())

# Answer 2
v1 = emp.columns.values
f1 = lambda x : x.lower()
emp.columns = list(map(f1, v1))

# 3) 전체 직원의 comm 평균

emp.comm.mean()    # 4명의 평균

# -- NA를 0으로 수정
# Answer 1
emp.comm = np.where(pd.isnull(emp.comm), 0, emp.comm)

# Answer 2
emp.comm = emp.comm.map(lambda x : 0 if pd.isnull(x) else x)

# Answer 3
emp.comm[pd.isnull(empcomm)] = 0

# 4) 7902 행 제거
emp.drop(7902, 0)

# 5) hiredate 컬럼을 hdate로 변경                                                 # *
emp.rename({'hiredate':'hdate'}, axis = 1)
emp.columns.values[emp.columns.values == 'hiredate'] = 'hdate'
# --------------------------------------------------------------------------- #

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 32.
# 1) 3 X 4 배열 생성 후 a,b,c,d 컬럼을 갖는 df1 데이터프레임 생성
a1 = np.arange(1,13).reshape(3,4)
df1 = DataFrame(a1, columns=['a','b','c','d'])

# 2) 2 X 4 배열 생성 후 a,b,c,d 컬럼을 갖는 df2 데이터프레임 생성
a2 = np.arange(1,9).reshape(2,4)
df2 = DataFrame(a2, columns=['a','b','c','d'])

# 3) 위 두 데이터프레임 union 후 df3 생성
df3 = df1.append(df2, ignore_index=True)

# 4) df3에서 0,2,4 행 선택해서 새로운 데이터 프레임 df4 생성
df4 = df3.iloc[[0,2,4],:]

# 5) df3에서 'b','d' 컬럼 선택 후 새로운 데이터 프레임 df5 선택
df5 = df3.loc[:,['b','d']]

# 6) df3 - df4 수행(NA 리턴 없이)                                                 # *
df3 - df4
df3 - df4.reindex(df3.index)  # NA를 포함하는 연산은 NA를 리턴
df3 - df4.reindex(df3.index).fillna(0)  # NA를 0으로 치환 후 연산
df3.sub(df4, fill_value = 0)

# 7) 다음의 데이터 프레임에서 2000년 기준 가격 상승률 출력
df1 = DataFrame({'2000':[1000,1100,1200],
                 '2001':[1150,1200,1400],
                 '2002':[1300,1250,1410]}, index = ['a','b','c'])

(1150 - 1000) / 1000 * 100                       # 스칼라 연산
(df1['2001'] - df1['2000']) / df1['2000'] * 100  # Series 연산

# 사칙연산 메서드 활용
df1.sub(df1['2000'], axis=0).div(df1['2000'], axis=0) * 100

# 행, 열 전치후 브로드캐스팅 연산(
 메서드 필요 X)
df2 = df1.T
((df2 - df2.loc['2000',:]) / df2.loc['2000',:] * 100).T
# --------------------------------------------------------------------------- #
reviewed 1 2020-09-17