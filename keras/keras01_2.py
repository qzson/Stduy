#1. 데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])
x2 = np.array([4,5,6])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(300))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
#5000개 시작으로 -3000 씩 레이어를 잡았다. (가장 결과가 좋았음 4.000008) // 1000개도 나쁘지 않음 4.000039 // 50, +100, -100 순차적 -> (4.0000105) 빠르다.
# 현재 노드 값이 제일 좋았다. 하지만 이것이 좋은 것이라고 장담할 수 없다.

#3. 훈련 (컴파일)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print("acc : ", acc)

y_predict = model.predict(x2) #변수하나 만들자. 
print(y_predict)

# 레이어, 에포치스를 수정을 통해 accuracy 값을 1에 가깝게 만드는 것이 중요.