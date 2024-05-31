테스트 셋에 대한 평가가 좋지 못한 상황
테스트 셋을 regular vs irregular로 분할 하여 각각에 대해 테스트 수행

### Split Application
이미지를 띄우고 ←, →키를 입력하여 regular, irregular를 구분하는 애플리케이션 제작
다음 이미지도 같이 보여주면 속도가 매우 빨라짐


※ regular, irregular 구분 기준 (사람이 자체적으로)
※ blured, curved, low Illuminance, irregular font 등의 경우 irregular로 분류


Regular: 4175장
Irregular: 1364장

### Result
Regular
acc:0.682
norm_edit_dis:0.842

Irregular
acc:0.330
norm_edit_dis:0.572
