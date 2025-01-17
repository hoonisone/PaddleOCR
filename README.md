ABINet_A1: Head는 하나이며 feature를 split, concat하며 각 자소별 특징 연산
ABINet_A2: 전체 Head를 추론마다 독립적으로 구성
ABINet_A3: Character를 중심으로 각 feature에 Utf8 head를 붙임 
ABINet_A4: Utf8을 중심으로 각 feature에 Character head를 붙임