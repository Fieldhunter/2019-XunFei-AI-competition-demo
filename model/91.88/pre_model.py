from keras.layers import Input, concatenate, Embedding, Dropout, add
from keras.layers.normalization import BatchNormalization
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

	def res_block(x, layer_size):
		drop1 = Dropout(dropout_ALPHA)(x)
		den1 = Dense(layer_size)(drop1)
		batch1 = BatchNormalization(epsilon=0.00001)(den1)
		rel1 = Activation('relu')(batch1)

		drop2 = Dropout(dropout_ALPHA - 0.1)(rel1)
		den2 = Dense(layer_size)(drop2)
		batch2 = BatchNormalization(epsilon=0.00001)(den2)

		return add([x, batch2])

	# 多输入
	input1 = Input(shape=(114,))
	input2 = Input(shape=(10,))

	# 分支部分
	dense1 = Dense(64)(input1)
	active1 = Activation('relu')(dense1)
	embedding = Embedding(18375, 128, input_length=(10,))(input2)
	flatten = Flatten()(embedding)

	# 合并
	merge = concatenate([active1, flatten])

	dropout1 = Dropout(dropout_ALPHA)(merge)
	dense2 = Dense(1024)(dropout1)
	active2 = Activation('relu')(dense2)

	dropout2 = Dropout(dropout_ALPHA)(active2)
	dense3 = Dense(512)(dropout2)
	active3 = Activation('relu')(dense3)

	res1 = res_block(active3, 512)
	active4 = Activation('relu')(res1)

	dropout3 = Dropout(dropout_ALPHA)(active4)
	dense4 = Dense(256)(dropout3)
	active5 = Activation('relu')(dense4)

	res2 = res_block(active5, 256)
	active6 = Activation('relu')(res2)

	dropout4 = Dropout(dropout_ALPHA)(active6)
	dense5 = Dense(128)(dropout4)
	active7 = Activation('relu')(dense5)

	res3 = res_block(active7, 128)
	active8 = Activation('relu')(res3)

	dropout6 = Dropout(dropout_ALPHA)(active8)
	dense6 = Dense(1)(dropout6)
	active9 = Activation('sigmoid')(dense6)

	model = Model(inputs=[input1,input2], outputs=active9)
	model.compile(loss=LOSS, metrics=[f1], optimizer=OPT)

	return model

