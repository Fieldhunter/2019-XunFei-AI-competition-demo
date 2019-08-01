from keras.layers import Input, concatenate, Embedding, Dropout
from keras.layers.core import Dense, Activation, Flatten
from keras.models import Model
from keras import backend as K

"""
	超参数:
		OPT = 'adam'
		LOSS = 'binary_crossentropy'
		dropout_ALPHA = 0.4
		BATCH_SIZE = 2048
		EPOCHS = 450
"""


def model(LOSS, OPT, dropout_ALPHA):
	# 计算f1指标
	def f1(y_true, y_pred):
		def recall(y_true, y_pred):
			true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
			possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
			recall = true_positives / (possible_positives + K.epsilon())

			return recall

		def precision(y_true, y_pred):
			true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
			predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
			precision = true_positives / (predicted_positives + K.epsilon())

			return precision

		precision = precision(y_true, y_pred)
		recall = recall(y_true, y_pred)

		return 2 * ((precision*recall) / (precision + recall + K.epsilon()))

	# 多输入
	input1 = Input(shape=(114,))
	input2 = Input(shape=(10,))

	# 分支部分
	dense1 = Dense(64)(input1)
	active1 = Activation('relu')(dense1)
	embedding = Embedding(18375, 512, input_length=(10,))(input2)
	flatten = Flatten()(embedding)

	# 合并
	merge = concatenate([active1, flatten])

	dropout1 = Dropout(dropout_ALPHA)(merge)
	dense2 = Dense(4096)(dropout1)
	active2 = Activation('relu')(dense2)

	dropout2 = Dropout(dropout_ALPHA)(active2)
	dense3 = Dense(4096)(dropout2)
	active3 = Activation('relu')(dense3)

	dropout3 = Dropout(dropout_ALPHA)(active3)
	dense4 = Dense(2048)(dropout3)
	active4 = Activation('relu')(dense4)

	dropout4 = Dropout(dropout_ALPHA)(active4)
	dense5 = Dense(2048)(dropout4)
	active5 = Activation('relu')(dense5)

	dropout5 = Dropout(dropout_ALPHA)(active5)
	dense6 = Dense(1024)(dropout5)
	active6 = Activation('relu')(dense6)

	dropout6 = Dropout(dropout_ALPHA)(active6)
	dense7 = Dense(1024)(dropout6)
	active7 = Activation('relu')(dense7)

	dropout7 = Dropout(dropout_ALPHA - 0.1)(active7)
	dense8 = Dense(512)(dropout7)
	active8 = Activation('relu')(dense8)

	dropout8 = Dropout(dropout_ALPHA - 0.1)(active8)
	dense9 = Dense(512)(dropout8)
	active9 = Activation('relu')(dense9)

	dropout9 = Dropout(dropout_ALPHA - 0.1)(active9)
	dense10 = Dense(256)(dropout9)
	active10 = Activation('relu')(dense10)

	dropout10 = Dropout(dropout_ALPHA - 0.1)(active10)
	dense11 = Dense(256)(dropout10)
	active11 = Activation('relu')(dense11)

	dense12 = Dense(1)(active11)
	active12 = Activation('sigmoid')(dense12)


	model = Model(inputs=[input1,input2], outputs=active12)
	model.compile(loss=LOSS, metrics=[f1], optimizer=OPT)

	return model
