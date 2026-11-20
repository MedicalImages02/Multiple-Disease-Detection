import tensorflow as tf
from keras.src.optimizers import Adam
from tensorflow.keras import layers, models, backend as K
import numpy as np

from Evaluation import evaluation


def novel_loss(y_true, y_pred):
    n = tf.cast(tf.size(y_true), tf.float32)
    mse = tf.reduce_sum(tf.square(y_true - y_pred))
    mae = tf.reduce_sum(tf.abs(y_true - y_pred))
    return -(2.0 / (n ** 2)) * mse + mae


def cross_covariance_attention(x, dim):
    q = layers.Dense(dim)(x)
    k = layers.Dense(dim)(x)
    v = layers.Dense(dim)(x)

    cov = tf.matmul(q, k, transpose_a=True)
    scale = tf.math.sqrt(tf.cast(dim, tf.float32))
    attn = tf.nn.softmax(cov / scale)

    return tf.matmul(attn, v)


def pyramid_encoder(input_shape, latent_dim):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inp)
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)

    x = layers.Reshape((-1, 128))(x)
    x = cross_covariance_attention(x, 128)
    x = layers.Flatten()(x)

    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)

    def sampling(args):
        z_mean, z_log_var = args
        eps = K.random_normal(shape=K.shape(z_mean))
        return z_mean + K.exp(0.5 * z_log_var) * eps

    z = layers.Lambda(sampling)([z_mean, z_log_var])

    return models.Model(inp, [z, z_mean, z_log_var], name="Encoder")


def decoder(latent_dim):
    inp = layers.Input(shape=(latent_dim,))
    x = layers.Dense(4 * 4 * 128, activation='relu')(inp)
    x = layers.Reshape((4, 4, 128))(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)

    out = layers.Conv2DTranspose(3, 3, padding='same', activation='sigmoid')(x)
    return models.Model(inp, out, name="Decoder")


def discriminator(sol, input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, strides=2, activation='relu')(inp)
    x = layers.Conv2D(128, 3, strides=2, activation='relu')(x)
    x = layers.Flatten()(x)
    out = layers.Dense(sol[0], activation='sigmoid')(x)
    return models.Model(inp, out, name="Discriminator")


def build_cc_pe_vgan(sol, input_shape, latent_dim):
    enc = pyramid_encoder(input_shape, latent_dim)
    dec = decoder(latent_dim)
    disc = discriminator(sol, input_shape)
    disc.trainable = False

    img = layers.Input(shape=input_shape)
    z, z_mean, z_log_var = enc(img)
    recon = dec(z)
    validity = disc(recon)

    model = models.Model(img, [recon, validity])

    kl_loss = -0.5 * K.mean(
        1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    )

    model.add_loss(kl_loss)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=sol[1]),
        loss=[novel_loss, 'binary_crossentropy']
    )

    return model


def AutoEn_Feat(Image, Target):
    IMG_SIZE = 256
    Train_X = np.zeros((Image.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Image.shape[0]):
        temp = np.resize(Image[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))
    model = build_cc_pe_vgan((Train_X.shape[1], Train_X.shape[2], Train_X.shape[3]), Target.shape[1])

    # Train
    model.fit(Train_X, Target,
        epochs=3,
        batch_size=8
    )
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions
    layerNo = 8
    Feats = []
    for i in range(Image.shape[0]):
        test = Image[i, :, :][np.newaxis, ...]
        layer_out = np.asarray(functors[layerNo]([test])).squeeze()
        Feats.append(layer_out)
    return np.asarray(Feats)


def Model_VAutoEn(Image, Target, sol=None):
    if sol is None:
        sol = [5, 0.01, 5]
    Feat = AutoEn_Feat(Image, Target)
    learnperc = round(Feat.shape[0] * 0.75)  # Split Training and Testing Datas
    Train_Data = Feat[:learnperc, :]
    Train_Target = Target[:learnperc, :]
    Test_Data = Feat[learnperc:, :]
    Test_Target = Target[learnperc:, :]
    X_train = Train_Data.reshape((Train_Data.shape[0], Train_Data.shape[1], 1, 1))
    X_test = Test_Data.reshape((Test_Data.shape[0], Test_Data.shape[1], 1, 1))
    model = build_cc_pe_vgan(sol, (X_train.shape[1], X_train.shape[2], X_train.shape[3]), Train_Target.shape[1])
    model.compile(optimizer=Adam(learning_rate=sol[1]),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    # Train
    model.fit(X_train, Target,
        epochs=sol[2],
        batch_size=8
    )
    pred = model.predict(X_test)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(Test_Target, pred)

    return Eval, pred