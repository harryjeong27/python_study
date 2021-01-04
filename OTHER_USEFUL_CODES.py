# Other Useful Codes

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

# 모듈 내 함수 목록 확인
dir(math)

# 작업 디렉토리
import os
os.getcwd()    # 작업 디렉토리 확인
os.chdir('/Users/harryjeong/DA_Practice')    # 작업 디렉토리 설정

# 파이썬 버젼확인
#주피터 노트북에서 파이썬 버전 확인하는 법
import sys
print("--sys.version—“)
print(sys.version)

# describe
df1.describe()
df1.describe(include = ['0'] # categorical features

# 2개 이상 표 한번에 보기
display(df1, df2)

# 항목별 개수
s1.value_counts()

# pickle
# 참고 : [[python] 파이썬 pickle 피클 활용법](https://korbillgates.tistory.com/173)
import pickle
with open('jeju_all.pickle', 'wb') as fw:
     pickle.dump(df_full, fw)

with open('jeju_all.pickle','rb') as fr:
    jeju_all = pickle.load(fr)

jeju_all.tail(3)

# glob
파일들의 리스트를 뽑을 때 사용 (파일의 경로 이용)
from glob import glob
glob('*.exe')               # 현재 디렉터리의 .exe 파일
# ['python.exe', 'pythonw.exe']
glob('*.txt')               # 현재 디렉터리의 .txt 파일
# ['LICENSE.txt', 'NEWS.txt']