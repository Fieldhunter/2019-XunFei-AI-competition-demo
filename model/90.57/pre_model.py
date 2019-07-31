from keras.layers import Input, concatenate, Embedding, Dropout
from keras.layers.core import Dense, Activation, Flatten
from keras.models import Model
from keras import backend as K


"""
	超参数:
		OPT = 'adam'
		LOSS = 'binary_crossentropy'
		dropout_ALPHA = 0.5
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
	dense1 = Dense(64, kernel_initializer="RandomNormal")(input1)
	active1 = Activation('relu')(dense1)
	embedding = Embedding(18375, 512, input_length=(10,), embeddings_initializer="RandomNormal")(input2)
	flatten = Flatten()(embedding)

	# 合并
	merge = concatenate([active1, flatten])

	dropout1 = Dropout(dropout_ALPHA)(merge)
	dense2 = Dense(4096, kernel_initializer="RandomNormal")(dropout1)
	active2 = Activation('relu')(dense2)

	dropout2 = Dropout(dropout_ALPHA)(active2)
	dense3 = Dense(2048, kernel_initializer="RandomNormal")(dropout2)
	active3 = Activation('relu')(dense3)

	dropout3 = Dropout(dropout_ALPHA)(active3)
	dense4 = Dense(2048, kernel_initializer="RandomNormal")(dropout3)
	active4 = Activation('relu')(dense4)

	dropout4 = Dropout(dropout_ALPHA)(active4)
	dense5 = Dense(1024, kernel_initializer="RandomNormal")(dropout4)
	active5 = Activation('relu')(dense5)

	dropout5 = Dropout(dropout_ALPHA)(active5)
	dense6 = Dense(1024, kernel_initializer="RandomNormal")(dropout5)
	active6 = Activation('relu')(dense6)

	dropout6 = Dropout(dropout_ALPHA)(active6)
	dense7 = Dense(512, kernel_initializer="RandomNormal")(dropout6)
	active7 = Activation('relu')(dense7)

	dropout7 = Dropout(dropout_ALPHA)(active7)
	dense8 = Dense(512, kernel_initializer="RandomNormal")(dropout7)
	active8 = Activation('relu')(dense8)

	dense9 = Dense(256, kernel_initializer="RandomNormal")(active8)
	active9 = Activation('relu')(dense9)

	dense10 = Dense(1, kernel_initializer="RandomNormal")(active9)
	active10 = Activation('sigmoid')(dense10)

	model = Model(inputs=[input1,input2], outputs=active10)
	model.compile(loss=LOSS, metrics=[f1], optimizer=OPT)

	return model
