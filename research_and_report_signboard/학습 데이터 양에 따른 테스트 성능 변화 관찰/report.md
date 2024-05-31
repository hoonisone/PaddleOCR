### WHY
* 학습데이터(수집한 용역 데이터)로 학습한 뒤 테스트 데이터(상무)에 대한 성능을 확인한 결과가 썩 좋지 못함
* 학습 데이터가 늘어날 수록, 특히 테스트 셋과 유사한 학습 데이터가 늘어날 수록 성능이 올라감을 보이고 싶음

### WHAT
* 테스트 셋 중 일부를 테스트로 사용하고 나머지는 학습 데이터로 추가하여 학습데이터가 늘어남에 따른 테스트 성능의 변하를 관찰

### HOW
##### Split
학습된 모델의 결과를 참고하여 각 세트에 대해 norm_edit의 평균이 서로 유사하도록 분할 함
1) norm_edit에 따라 구간별로 split (0.1, 0,2, ... 1.0)
2) 앞에서 부터 차례로 하나씩 set를 번갈아가며 샘플링
3) 평균 norm_edit 확인 결과 오차 매우 작음

##### Task
테스트 용으로 하위 셋 하나를 제외하고 나머지 sub set에 대해
1개, 2개, ... , n-1개 사용한 경우에 대해 모델을 학습한 뒤 테스트 성능 확인


### Experiment Detail
분할 개수(n) = 4
set 0 => sample num = 1385, avarage_norm_dis = 0.22915457372309336 
set 1 => sample num = 1385, avarage_norm_dis = 0.22896338093037533 
set 2 => sample num = 1385, avarage_norm_dis = 0.22856639781548307 
set 3 => sample num = 1384, avarage_norm_dis = 0.22918631388064212 

### Performance


### Result
* Performance
    |  | Acc | Norm_edit |
    |-----|-----|-----|
    | set x 1 | 0.670 | 0.841 |
    | set x 1 | 0.696 | 0.851 |
    | set x 1 | 0.714 | 0.865 |