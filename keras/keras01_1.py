#1. 데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim = 1)) # 인풋, 아웃풋 아래는 순차적으로 되기 때문에 생략.
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 훈련 (컴파일)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) #mse 이라는 걸 쓴다.(계산을 잘하기 위해) loss는 손실, 그거에 대한 최적화를 위해 adam을 쓴다. metrics 애큐러시로 훈련 과정을 보여주는 것
model.fit(x, y, epochs=100, batch_size=1) #fit - x, y 넣어서 훈련시킨다, epochs 훈련을 몇 번 시키는지, batch_size 몇 개씩 짤라서 작업시키냐

#4. 평가 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print("acc : ", acc) #acc - 변수, x와 y를 평가했을 때

#accuracy 1.0 이 나왔다는건 정확도 100이지만, 실제론 틀린 것. 1,2,3 을 넣었을 때 1,2,3이 나왔다는 뜻

