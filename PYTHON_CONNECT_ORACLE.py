# 파이썬 - 오라클 연동
# 1. oracle_client 혹은 oracle 설치
# ** path 설정 필요(시스템변수, 사용자변수 둘 다)

# 2. cx_oracle 설치
# OS 명령어 -> 터미널에서 수행 -> 오류 -> 파이썬에서 수행하면 됨
conda install -c https://conda.anaconda.org/anaconda cx_oracle    

# 3. oracle 연동
# 3-1) 모듈 로딩
import cx_Oracle

# 3-2) 오라클 연결
# 설치되어 있는 OS의 IP
con = cx_Oracle.connect("system/oracle@172.30.101.19:1521/orcl")    # Error
con = cx_Oracle.connect('system', 'oracle', '172.30.101.19/orcl')   # Error

# 3-3) sql 실행 및 데이터 불러오기
import pandas as pd
df_emp = pd.read_sql("select * from emp",con=con)

# 3-4) 기타 화면에 출력하는 방법
cur = con.cursor()
cur.execute("select * from emp")

for row in cur:
	print(row)
cur.close()
con.close()

# ** 한글 깨짐 (connection을 다시 맺을 필요 있음)
import os
os.putenv('NLS_LANG', 'KOREAN_KOREA.KO16MSWIN949') 