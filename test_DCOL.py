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

    model1.load_weights(save_dir1+get_model_nameE(fold_var))
    filenames = test_generator.filenames
    nb_samples = len(filenames)
    predict =model1.predict(test_generator,steps = nb_samples)
    y_pred = np.round(predict)
    y_true = test_df['labels'].astype('float32')
    y_true = np.array(y_true)

    y_pred[y_pred < 2] = 0
    y_pred[(y_pred >= 2) & (y_pred < 6)] = 1
    y_pred[y_pred >= 6] = 2

    y_true[y_true < 2] = 0
    y_true[(y_true >= 2) & (y_true < 6)] = 1
    y_true[y_true >= 6] = 2

    matrix = confusion_matrix(y_true, y_pred)
    print(matrix)

    A = evaluate_multiclass(y_true, y_pred)
    print('Accuracy=')
    print(A[0], Average(A[0]))
    print('PPV=')
    print(A[1], Average(A[1]))
    print('NPV=')
    print(A[2], Average(A[2]))
    print('Sensitivity')
    print(A[3], Average(A[3]))
    print('Specificity')
    print(A[4], Average(A[4]))
    print('F1-Score')
    print(A[6], Average(A[6]))

fold_var += 1

