import keras
import numpy as np
from pathlib import Path
import pandas as pd
import librosa
import sklearn.metrics
import tensorflow as tf
from keras import backend as K

def get_data(DATA):
    CSV_TRN_CURATED = DATA/'train_curated.csv'
    CSV_TRN_NOISY = DATA/'train_noisy.csv'
    CSV_SUBMISSION = DATA/'sample_submission.csv'

    TRN_CURATED = DATA/'train_curated'
    TRN_NOISY = DATA/'train_noisy'
    TEST = DATA/'test'
    trn_curated_df = pd.read_csv(CSV_TRN_CURATED)
    trn_noisy_df = pd.read_csv(CSV_TRN_NOISY)
    test_df = pd.read_csv(CSV_SUBMISSION)
    categories = test_df.columns[1:].values

    y_trn_curated = np.vstack(trn_curated_df['labels'].apply(lambda x: x.split(',')).apply(lambda x: np.isin(categories, x))).astype(int)
    X_trn_curated = trn_curated_df['fname'].values

    for i, X in enumerate(X_trn_curated):
        X_trn_curated[i] = TRN_CURATED/X
    
    y_trn_noisy = np.vstack(trn_noisy_df['labels'].apply(lambda x: x.split(',')).apply(lambda x: np.isin(categories, x))).astype(int)
    X_trn_noisy = trn_noisy_df['fname'].values

    for i, X in enumerate(X_trn_noisy):
        X_trn_noisy[i] = TRN_NOISY/X
        
    X_test = test_df['fname'].values

    for i, X in enumerate(X_test):
        X_test[i] = TEST/X
    return X_trn_curated, y_trn_curated, X_trn_noisy, y_trn_noisy, X_test, test_df, categories

class DataGeneratorPreproc(keras.utils.Sequence):
    def __init__(self, folder, audio_duration=4, batch_size=32, 
                 shuffle=True, random_seed=42, filelist=None, augment=1,
                 sample_rate=44100,  hop_length=1024, repeat=False):
        np.random.seed(random_seed)
        self.augment = augment
        self.batch_size = batch_size
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.shuffle = shuffle
        self.repeat = repeat
        if filelist is None:
            self.filelist = list(folder.glob('*'))
        else:
            self.filelist = filelist
        self.on_epoch_end()
        return

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.filelist) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        Xs = []
        ys = []
        audio_duration = self.audio_duration
        if audio_duration is None:
            audio_duration = 4+np.random.rand()*6
        
        image_width = int(audio_duration*self.sample_rate/self.hop_length)
        for i, index in enumerate(indexes):
            X, y = np.load(self.filelist[index], allow_pickle=True)
            padding = image_width - X.shape[1]
            # print(i, padding)
            if self.augment == 0:
                if padding>0:
                    if self.repeat:
                        X_rep = np.hstack([X for i in range(int(np.ceil((image_width + padding) / X.shape[1])))] )
                        Xc = X_rep[:, : image_width]
                    else:
                        Xc = np.hstack([X, np.zeros((X.shape[0], padding))])
                    Xs.append(Xc[:, :image_width])
                    ys.append(y)
                else:
                    Xs.append(X[:, :image_width])
                    ys.append(y)
            elif padding>0:
                for N in range(self.augment):
                    if self.repeat:
                        X_rep = np.hstack([X for i in range(int(np.ceil((image_width + padding) / X.shape[1])))] )
                        rand_padding = np.random.randint(padding)
                        Xc = X_rep[:, rand_padding: image_width+rand_padding]
                    else:
                        rand_padding = np.random.randint(padding)
                        Xc = np.hstack([np.zeros((X.shape[0], rand_padding)), X, np.zeros((X.shape[0], padding - rand_padding))])
                    Xs.append(Xc[:, :image_width])
                    ys.append(y)
                    
            elif padding<0:
                for N in range(self.augment):
                    Xc = X[:, np.random.randint(-padding):]
                    Xs.append(Xc[:, :image_width])
                    ys.append(y)
            else:
                Xs.append(X[:, :image_width])
                ys.append(y)
        Xs = np.array(Xs)
        return Xs.reshape( Xs.shape +  (1,)), np.array(ys)
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.filelist))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

class DataGeneratorMel(keras.utils.Sequence):
    @staticmethod
    def get_audio(index, X, folder, sr = 44100, duration=2, hop_length=1024, n_fft=2048):
        if folder is not None:
            audio_file = folder/X[index]
        else:
            audio_file = X[index]
        # Open file with original sample rate
        audio_samples, orig_sr = librosa.load(audio_file, sr=None)
        
        if duration is not None:
            stop = int(duration*orig_sr)
            X_truncated = audio_samples[0:stop]
            X_truncated = np.hstack([X_truncated, np.zeros(stop-len(X_truncated))])
        else:
            X_truncated = audio_samples
        mel_spec = librosa.feature.melspectrogram(X_truncated, sr=sr, hop_length=hop_length, n_fft=n_fft)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db

    @staticmethod
    def get_batch(indexes, X, y, folder, sr = 44100, duration=2, hop_length=1024, n_fft=2048):
        batch_X = []
        batch_y = np.zeros([0, y.shape[1]])
        for i in indexes:
            if duration==0:
                duration = 2+np.random.rand()*12
            X_ = DataGeneratorMel.get_audio(i, X, folder, sr = sr, duration=duration, hop_length=hop_length, n_fft=n_fft)
            batch_X.append(X_)
            # batch_X = np.vstack([batch_X, X_])
            batch_y = np.vstack([batch_y, y[i]])
        batch_X = np.array(batch_X)
        return batch_X, batch_y
    
    def __init__(self, X_train, y_train, audio_folder, batch_size=128, 
                 shuffle=True, random_seed=42, 
                 sample_rate=44100, audio_duration=2,  hop_length=1024, n_fft=2048, TwoD=True):
        np.random.seed(random_seed)
        self.batch_size = batch_size
        self.audio_folder = audio_folder
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.X_train = X_train
        self.y_train = y_train
        self.shuffle = shuffle
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.on_epoch_end()
        self.TwoD = TwoD
        return

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = DataGeneratorMel.get_batch(indexes, self.X_train, self.y_train, self.audio_folder, 
                                       self.sample_rate, self.audio_duration, hop_length=self.hop_length, n_fft=self.n_fft)
        if self.TwoD:
            return X.reshape(*X.shape , 1), y
        else:
            return X, y
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X_train))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

class DataGenerator(keras.utils.Sequence):
    @staticmethod
    def get_audio(index, X, folder, sr = 44100, max_pre=1, duration=2, res_type='scipy', max_region=False):
        if folder is not None:
            audio_file = folder/X[index]
        else:
            audio_file = X[index]
        # subsample
        audio_samples, orig_sr = librosa.load(audio_file, sr=None)
        if max_region:
            argmax = np.argmax(np.abs(audio_samples))
            start = np.max([argmax - int(orig_sr*max_pre), 0])
            start = int(np.random.rand()*start)
        else:
            start = 0
        stop = start+duration*orig_sr
        X_truncated = audio_samples[start:stop]
        X_truncated = np.hstack([X_truncated, np.zeros(duration*orig_sr-len(X_truncated))])
        X_truncated = librosa.resample(X_truncated, orig_sr, sr, res_type=res_type)
        return (np.random.randint(2)*2-1) * X_truncated

    @staticmethod
    def get_batch(indexes, X, y, folder, sr = 44100, max_pre=1, duration=2, res_type='scipy'):
        batch_X = np.zeros([0, duration*sr])
        batch_y = np.zeros([0, y.shape[1]])
        for i in indexes:
            X_ = DataGenerator.get_audio(i, X, folder, sr = sr, max_pre=max_pre, duration=duration, res_type=res_type)
            batch_X = np.vstack([batch_X, X_])
            batch_y = np.vstack([batch_y, y[i]])
        return batch_X, batch_y
    
    def __init__(self, X_train, y_train, audio_folder, batch_size=128, 
                 shuffle=True, random_seed=42, 
                 sample_rate=11025, audio_duration=2, look_behind=0.5, sample_type='scipy'):
        np.random.seed(random_seed)
        self.batch_size = batch_size
        self.audio_folder = audio_folder
        self.sample_rate = sample_rate
        self.look_behind = look_behind
        self.audio_duration = audio_duration
        self.X_train = X_train
        self.y_train = y_train
        self.shuffle = shuffle
        self.sample_type = sample_type
        self.on_epoch_end()
        return

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = DataGenerator.get_batch(indexes, self.X_train, self.y_train, self.audio_folder, 
                                       self.sample_rate, self.look_behind, self.audio_duration, self.sample_type)
        return X.reshape(len(indexes), -1 , 1), y
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X_train))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
from keras.layers import Conv1D, MaxPool1D, Flatten, Dense, GlobalAveragePooling1D, BatchNormalization, Activation, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam

def simple_CNN_2D(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(filters=16, kernel_size=3, strides=1, activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(80, activation='linear'))
    return model

def simple_CNN_BN_2D(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=3, strides=1, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(filters=16, kernel_size=3, strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(80, activation='linear'))
    return model

def simple_CNN(dataGen):
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, strides=1, activation='relu', input_shape=(dataGen.sample_rate*dataGen.audio_duration,1, )))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=16, kernel_size=3, strides=1, activation='relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(80, activation='linear'))
    return model

def simple_CNN_BN(dataGen):
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, strides=1, input_shape=(dataGen.sample_rate*dataGen.audio_duration,1, )))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=16, kernel_size=3, strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(80, activation='linear'))
    return model

def get_simil_yolo(dataGen):
    unit_shape = 1024
    # This is the window that at the end overlaps given input shape
    time_unit = unit_shape/dataGen.sample_rate
    # print(time_unit)
    input_size = dataGen.audio_duration*dataGen.sample_rate
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, strides=1, activation='relu', padding='same',
                     input_shape=(input_size,1, )))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=3, strides=1, activation='relu', padding='same'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=1024, kernel_size=3, strides=1, activation='relu', padding='same'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=3, strides=1, activation='relu', padding='same'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=80, kernel_size=3, strides=1, activation='linear', padding='same'))
    model.add(MaxPool1D(pool_size=2))
    model.add(GlobalAveragePooling1D())
    return model

def get_simil_yolo_BN(dataGen):
    unit_shape = 1024
    # This is the window that at the end overlaps given input shape
    time_unit = unit_shape/dataGen.sample_rate
    # print(time_unit)
    input_size = dataGen.audio_duration*dataGen.sample_rate
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, strides=1, padding='same', input_shape=(input_size,1, )))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=512, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=1024, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=512, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=80, kernel_size=3, strides=1, activation='linear', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=2))
    model.add(GlobalAveragePooling1D())
    return model

# Metrics

def calculate_overall_lwlrap_sklearn(truth, scores):
    """Calculate the overall lwlrap using sklearn.metrics.lrap."""
    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(
      truth[nonzero_weight_sample_indices, :] > 0, 
      scores[nonzero_weight_sample_indices, :], 
      sample_weight=sample_weight[nonzero_weight_sample_indices])
    return overall_lwlrap

def tf_lwlrap_2(y_true, y_pred):
    out = tf.py_func(calculate_overall_lwlrap_sklearn, (y_true, y_pred), tf.double)
    return out

def tf_one_sample_positive_class_precisions(y_true, y_pred) :
    num_samples, num_classes = y_pred.shape
    
    # find true labels
    pos_class_indices = tf.where(y_true > 0) 
    
    # put rank on each element
    retrieved_classes = tf.nn.top_k(y_pred, k=num_classes).indices
    sample_range = tf.zeros(shape=tf.shape(tf.transpose(y_pred)), dtype=tf.int32)
    sample_range = tf.add(sample_range, tf.range(tf.shape(y_pred)[0], delta=1))
    sample_range = tf.transpose(sample_range)
    sample_range = tf.reshape(sample_range, (-1,num_classes*tf.shape(y_pred)[0]))
    retrieved_classes = tf.reshape(retrieved_classes, (-1,num_classes*tf.shape(y_pred)[0]))
    retrieved_class_map = tf.concat((sample_range, retrieved_classes), axis=0)
    retrieved_class_map = tf.transpose(retrieved_class_map)
    retrieved_class_map = tf.reshape(retrieved_class_map, (tf.shape(y_pred)[0], num_classes, 2))
    
    class_range = tf.zeros(shape=tf.shape(y_pred), dtype=tf.int32)
    class_range = tf.add(class_range, tf.range(num_classes, delta=1))
    
    class_rankings = tf.scatter_nd(retrieved_class_map,
                                          class_range,
                                          tf.shape(y_pred))
    
    #pick_up ranks
    num_correct_until_correct = tf.gather_nd(class_rankings, pos_class_indices)

    # add one for division for "presicion_at_hits"
    num_correct_until_correct_one = tf.add(num_correct_until_correct, 1) 
    num_correct_until_correct_one = tf.cast(num_correct_until_correct_one, tf.float32)
    
    # generate tensor [num_sample, predict_rank], 
    # top-N predicted elements have flag, N is the number of positive for each sample.
    sample_label = pos_class_indices[:, 0]   
    sample_label = tf.reshape(sample_label, (-1, 1))
    sample_label = tf.cast(sample_label, tf.int32)
    
    num_correct_until_correct = tf.reshape(num_correct_until_correct, (-1, 1))
    retrieved_class_true_position = tf.concat((sample_label, 
                                               num_correct_until_correct), axis=1)
    retrieved_pos = tf.ones(shape=tf.shape(retrieved_class_true_position)[0], dtype=tf.int32)
    retrieved_class_true = tf.scatter_nd(retrieved_class_true_position, 
                                         retrieved_pos, 
                                         tf.shape(y_pred))
    # cumulate predict_rank
    retrieved_cumulative_hits = tf.cumsum(retrieved_class_true, axis=1)

    # find positive position
    pos_ret_indices = tf.where(retrieved_class_true > 0)

    # find cumulative hits
    correct_rank = tf.gather_nd(retrieved_cumulative_hits, pos_ret_indices)  
    correct_rank = tf.cast(correct_rank, tf.float32)

    # compute presicion
    precision_at_hits = tf.truediv(correct_rank, num_correct_until_correct_one)

    return pos_class_indices, precision_at_hits

def tf_lwlrap(y_true, y_pred):
    num_samples, num_classes = y_pred.shape
    pos_class_indices, precision_at_hits = (tf_one_sample_positive_class_precisions(y_true, y_pred))
    pos_flgs = tf.cast(y_true > 0, tf.int32)
    labels_per_class = tf.reduce_sum(pos_flgs, axis=0)
    weight_per_class = tf.truediv(tf.cast(labels_per_class, tf.float32),
                                  tf.cast(tf.reduce_sum(labels_per_class), tf.float32))
    sum_precisions_by_classes = tf.zeros(shape=(num_classes), dtype=tf.float32)  
    class_label = pos_class_indices[:,1]
    sum_precisions_by_classes = tf.unsorted_segment_sum(precision_at_hits,
                                                        class_label,
                                                       num_classes)
    labels_per_class = tf.cast(labels_per_class, tf.float32)
    labels_per_class = tf.add(labels_per_class, 1e-7)
    per_class_lwlrap = tf.truediv(sum_precisions_by_classes,
                                  tf.cast(labels_per_class, tf.float32))
    out = tf.cast(tf.tensordot(per_class_lwlrap, weight_per_class, axes=1), dtype=tf.float32)
    return out

def BCEwithLogits(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred, from_logits=True), axis=-1)