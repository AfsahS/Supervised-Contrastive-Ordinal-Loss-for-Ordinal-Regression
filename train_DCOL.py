from LCL_GCL_module import global_enc_proj, local_enc_proj
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from Losses import *
from DCOL_dataloader import DCMDataFrameIterator



save_dir = '/FastData/'

##### Encoder Projector weights ###
def get_model_nameA(k):
    return 'Global_SCOL'+str(k)+'.h5'
def get_model_nameC(k):
    return 'Local_SCOL_' + str(k) + '.h5'
### Regression Network/ To check individual performance of  Glbobal Module######
def get_model_nameB(k):
    return 'Global_SCOL_Class_'+str(k)+'.h5'
def get_model_nameD(k):
    return 'Local_SCOL_Class' + str(k) + '.h5'
def get_model_nameE(k):
    return 'DCOL' + str(k) + '.h5'


# augmentation parameters
train_augmentation_parameters = dict(
    rotation_range=15,
    shear_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    fill_mode='constant',
    cval=0)

test_augmentation_parameters = dict(
    rescale=0.0,
)
# training parameters
BATCH_SIZE = 16
CLASS_MODE = 'raw'
COLOR_MODE = 'rgb'
TARGET_SIZE = (320, 320)
SEED = 7

train_consts = {
    'seed': SEED,
    'batch_size': BATCH_SIZE,
    'class_mode': CLASS_MODE,
    'color_mode': COLOR_MODE,
    'target_size': TARGET_SIZE,
    'subset': 'training'

}
test_consts = {
    'batch_size': 1,  # should be 1 in testing
    'class_mode': CLASS_MODE,
    'color_mode': COLOR_MODE,
    'target_size': TARGET_SIZE,  # resize input images
    'shuffle': False
}
############### Train GLobal Contrastive Learning Module #######################
fold_var = 1
Kkfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=3)
for train_index, val_index in Kkfold.split(np.zeros(1914),Y):
    train_df = df.iloc[train_index]
    test_df = df.iloc[val_index]

    train_df, valid_df = train_test_split(train_df, test_size=0.2)
    train_data_generator = DCMDataFrameIterator(dataframe=train_df,
                              x_col='fileName',
                              y_col='labels',
                              image_data_generator=train_augmenter,
                              **train_consts, shuffle = True)
    valid_data_generator  = DCMDataFrameIterator(dataframe=valid_df,
                              x_col='fileName',
                              y_col='labels',
                              image_data_generator=test_augmenter,
                              **test_consts)
    test_generator = DCMDataFrameIterator(dataframe=test_df,
                                                x_col='fileName',
                                                y_col='labels',
                                                image_data_generator=test_augmenter,
                                                **test_consts)

    filenames = test_generator.filenames
    nb_samples = len(filenames)

    ###################### Global  COntarstive Module Training ################
    model = global_enc_proj()
    # model.load_weights(save_dir+get_model_nameA(fold_var))
    mC = tf.keras.Model(inputs = model.input, outputs = model.layers[-4].output) ### Discard Projection Layers
    mC.trainable = False
    in_fea = mC.output
    features = Dense(1280, activation="relu",)(in_fea)
    features = Dropout(0.4)(features)
    features = Dense(128, activation="relu",)(features)
    features = Dropout(0.2)(features)
    outputs = Dense(1, activation="linear")(features)
    model1 = keras.Model(inputs=mC.input, outputs=outputs)
    model1.load_weights(save_dir + get_model_nameB(fold_var))
    mcB = tf.keras.Model(inputs=model1.input, outputs=model1.layers[-6].output)
    pred_fea1 = mcB.output

    ###################### Local COntarstive Module Training ################
    model2 = local_enc_proj()
    # model.load_weights(save_dir + get_model_nameC(fold_var))
    mD = tf.keras.Model(inputs=model2.input, outputs=model2.layers[-4].output)
    mD.trainable =False
    in_fea = mD.output
    features = Dense(1280, activation="relu", name='d_1')(in_fea)
    features = Dropout(0.4)(features)
    features = Dense(128, activation="relu", name='d_2')(features)
    features = Dropout(0.2)(features)
    outputs = Dense(1, activation="linear", name='final_output')(features)
    model1 = keras.Model(inputs=mD.input, outputs=outputs)
    model1.load_weights(save_dir + get_model_nameD(fold_var))
    mcC = tf.keras.Model(inputs=model1.input, outputs=model1.layers[-6].output)
    pred_fea2 = mcC.output

    ##################### Fuse weights of Local and Global Module ##################
    pred_fea = tf.keras.layers.Concatenate(axis=-1)([pred_fea1, pred_fea2])

    features = Dense(1280, activation="relu", )(pred_fea)
    features = Dropout(0.4)(features)
    features = Dense(128, activation="relu", )(features)
    features = Dropout(0.2)(features)
    outputs = Dense(1, activation="linear")(features)
    merged = keras.Model(inputs=[model.input, model2.input], outputs=outputs)

    for i, w in enumerate(merged .weights):
        split_name = w.name.split('/')
        new_name = split_name[0] + '_' + str(i) + '/' + split_name[1] + '_' + str(i)
        merged .weights[i]._handle_name = new_name

    merged .compile(
        optimizer=optimizers.RMSprop(learning_rate=3e-4),
        loss=root_mean_squared_error,
        metrics=['mean_squared_error', pearson_r],
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                  patience=10, min_lr=1e-7)
    checkpoint = keras.callbacks.ModelCheckpoint(save_dir1 + get_model_nameE(fold_var),
                                                 monitor='val_loss', verbose=1,
                                                 save_best_only=True, mode='min')
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    callbacks_list = [reduce_lr, es, checkpoint]
    history = merged .fit(train_data_generator,
                              validation_data=valid_data_generator,
                              epochs=200,
                              verbose=2, callbacks=callbacks_list
                              )

fold_var += 1

