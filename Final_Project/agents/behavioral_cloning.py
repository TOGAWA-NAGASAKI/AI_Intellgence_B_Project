import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam

class BehavioralCloningAgent:
    def __init__(self, observation_space, action_space, learning_rate=0.01):
        # 注意：这里默认学习率改为了 0.01，有助于快速收敛
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        
        self.model = self._build_model()

    def _build_model(self):
        inputs = Input(shape=(self.observation_space,))
        
        # 简单的两层全连接网络
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        
        outputs = Dense(self.action_space, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # 显式指定 loss 函数，避免字符串引用可能带来的歧义
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        return model

    def train(self, states, actions, batch_size=32, epochs=50, validation_split=0.2):
        print(f"Start Training: Batch={batch_size}, Epochs={epochs}")
        history = self.model.fit(
            states, 
            actions, 
            batch_size=batch_size, 
            epochs=epochs, 
            validation_split=validation_split,
            verbose=1
        )
        return history

    def act(self, state):
        state = state.reshape(1, -1)
        # 使用 verbose=0 静默预测
        probs = self.model.predict(state, verbose=0)[0]
        return np.argmax(probs)

    def save_model(self, filepath):
        # 自动创建目录
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_weights(f"{filepath}_bc.weights.h5")
        print(f"Model saved to {filepath}_bc.weights.h5")

    def load_model(self, filepath):
        self.model.load_weights(f"{filepath}_bc.weights.h5")
        print(f"Model loaded from {filepath}_bc.weights.h5")