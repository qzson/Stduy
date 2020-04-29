#1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x2 = np.array([4,5,6])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(250))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#3. 훈련 (컴파일)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=100)

#4. 평가 예측
loss, acc = model.evaluate(x, y)
print("acc : ", acc)

y_predict = model.predict(x2)
print(y_predict)

# 데이터가 많아질수록 더 정확해진다.
# loss, optimizer 이런 것들은 하이퍼 파라미터라 부른다. 그런 것들을 수정하는 것이 타이퍼 파라미터 튜닝이다.
# 취미 하이퍼 파라미터 튜닝, 특기는 데이터 전처리
# 뱃치사이즈 삭제해도 돌아간다는 건.. 뱃치사이즈에 대한 기본값이 있다는 것.
# batch size 의 기본 값은 32
# 레이어는 7개 1,10,50,250,50,10,1 로 구성했을 때 나쁘지 않다.