from __future__ import print_function

from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Input, Dense, Dropout, Flatten, Activation, concatenate
from keras.models import Model

from src import config
from src.Hybrid import utils


def construct_model(model_type, input_shape=(None, None, 1)):
    inputs = Input(name='the_input', shape=input_shape, dtype='float32')
    # --------------------------------------------------------------------------#
    inner = Conv2D(filters=32,
                   kernel_size=(3, 3),
                   activation='relu',
                   input_shape=input_shape,
                   padding='Same',
                   name='Common_conv_1')(inputs)
    inner = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='Same', name='Common_conv_2')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='Common_pooling_1')(inner)
    inner = Dropout(config.drop_out, name='Common_pool_dropout_1')(inner)
    # --------------------------------------------------------------------------#
    inner = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='Same', name='Common_conv_3')(inner)
    inner = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='Same', name='Common_conv_4')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='Common_pooling_2')(inner)
    inner = Dropout(config.drop_out, name='Common_pool_dropout_2')(inner)
    # --------------------------------------------------------------------------#
    inner = Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='relu', name='Common_conv_5')(inner)
    branch_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu', name='Common_conv_6')(inner)
    if model_type == 'Hybrid' or model_type == 'OCR':
        inner = MaxPooling2D(pool_size=(2, 2), name='OCR_pooling_1')(branch_1)
        inner = Dropout(config.drop_out, name='OCR_pool_dropout_1')(inner)
        # # --------------------------------------------------------------------------#
        outlist = []
        outdict = {}
        for i in range(config.num_digits):
            name = 'digit_' + str(i + 1)
            outdict[name] = Conv2D(filters=32,
                                   kernel_size=(3, 3),
                                   padding='Same',
                                   activation='relu',
                                   name='OCR_digit_{}_conv_1'.format(i+1))(inner)
            outdict[name] = Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   padding='Same',
                                   activation='relu',
                                   name='OCR_digit_{}_conv_2'.format(i+1))(outdict[name])
            outdict[name] = Flatten(name='OCR_digit_{}_flatten'.format(i+1))(outdict[name])
            outdict[name] = Dense(512, activation='relu', name='OCR_digit_{}_dense_1'.format(i+1))(outdict[name])
            outdict[name] = Dropout(config.drop_out, name='OCR_digit_{}_dropout_1'.format(i+1))(outdict[name])
            outdict[name] = Dense(256, activation='relu', name='OCR_digit_{}_dense_2'.format(i+1))(outdict[name])
            outdict[name] = Dropout(config.drop_out, name='OCR_digit_{}_dropout_2'.format(i+1))(outdict[name])
            outdict[name] = Dense(128, activation='relu', name='OCR_digit_{}_dense_3'.format(i+1))(outdict[name])
            outdict[name] = Dropout(config.drop_out, name='OCR_digit_{}_dropout_3'.format(i+1))(outdict[name])
            if i == 2:
                outdict[name] = Dense(len(utils.char_vector), name='OCR_digit_{}_dense_4'.format(i+1))(outdict[name])
            else:
                outdict[name] = Dense(len(utils.digit_vector), name='OCR_digit_{}_dense_4'.format(i+1))(outdict[name])
            outdict[name] = Activation('sigmoid', name='OCR_digit_{}'.format(i+1))(outdict[name])
            outlist.append(outdict[name])

        output_ocr = concatenate(outlist, name='OCR')
    # --------------------------------------------------------------------------#
    if model_type == 'Hybrid' or model_type == 'Segmentation':
        inner_seg = Conv2DTranspose(filters=64, kernel_size=(3, 3),
                                    strides=(2, 2), padding='Same',
                                    activation='relu', name='Segmentation_convT_1')(branch_1)
        inner_seg = Conv2D(filters=64, kernel_size=(3, 3),
                           strides=(1, 1), padding='Same',
                           activation='relu', name='Segmentation_conv_1')(inner_seg)
        inner_seg = Conv2D(filters=32, kernel_size=(3, 3),
                           strides=(1, 1), padding='Same',
                           activation='relu', name='Segmentation_conv_2')(inner_seg)
        inner_seg = Dropout(config.drop_out, name='Segmentation_droupout_1')(inner_seg)
        # --------------------------------------------------------------------------#
        inner_seg = Conv2DTranspose(filters=64, kernel_size=(3, 3),
                                    strides=(2, 2), padding='Same',
                                    activation='relu', name='Segmentation_convT_2')(inner_seg)
        inner_seg = Conv2D(filters=64, kernel_size=(3, 3),
                           strides=(1, 1), padding='Same',
                           activation='relu', name='Segmentation_conv_3')(inner_seg)
        inner_seg = Conv2D(filters=32, kernel_size=(3, 3),
                           strides=(1, 1), padding='Same',
                           activation='relu', name='Segmentation_conv_4')(inner_seg)
        inner_seg = Dropout(config.drop_out, name='Segmentation_dropout_2')(inner_seg)
        # --------------------------------------------------------------------------#
        if model_type == 'Segmentation':
            output_seg = Conv2D(filters=1, kernel_size=(3, 3),
                                strides=(1, 1), padding='Same',
                                activation='sigmoid', name='Segmentation_conv_5')(inner_seg)
        else:
            inner_seg = Conv2D(filters=1, kernel_size=(3, 3),
                               strides=(1, 1), padding='Same',
                               activation='sigmoid', name='Segmentation_conv_5')(inner_seg)
            output_seg = Flatten(name='Segmentation')(inner_seg)

    if model_type == 'Hybrid':
        return Model(inputs=inputs, outputs=[output_ocr, output_seg])
    elif model_type == 'OCR':
        return Model(inputs=inputs, outputs=output_ocr)
    elif model_type == 'Segmentation':
        return Model(inputs=inputs, outputs=output_seg)
