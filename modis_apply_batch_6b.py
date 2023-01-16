import pickle
import netCDF4 as nc
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import sys
import os


def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def model_application(data_dir, modis_date, save_dir, save_name):

    Modis_rrs_412 = os.path.join(data_dir, modis_date) + '.L3m_MO_RRS_Rrs_412_4km.nc'
    Modis_rrs_443 = os.path.join(data_dir, modis_date) + '.L3m_MO_RRS_Rrs_443_4km.nc'
    Modis_rrs_488 = os.path.join(data_dir, modis_date) + '.L3m_MO_RRS_Rrs_488_4km.nc'
    Modis_rrs_531 = os.path.join(data_dir, modis_date) + '.L3m_MO_RRS_Rrs_531_4km.nc'
    Modis_rrs_547 = os.path.join(data_dir, modis_date) + '.L3m_MO_RRS_Rrs_547_4km.nc'
    Modis_rrs_667 = os.path.join(data_dir, modis_date) + '.L3m_MO_RRS_Rrs_667_4km.nc'

    f_412 = nc.Dataset(Modis_rrs_412)  # Rrs
    f_443 = nc.Dataset(Modis_rrs_443)  # Rrs
    f_488 = nc.Dataset(Modis_rrs_488)
    f_531 = nc.Dataset(Modis_rrs_531)
    f_547 = nc.Dataset(Modis_rrs_547)
    f_667 = nc.Dataset(Modis_rrs_667)

    # read data
    lat = np.array(f_443['lat'])  # global
    lon = np.array(f_443['lon'])  # global

    Rrs412 = np.array(f_412['Rrs_412'])
    Rrs412[Rrs412 < 0] = np.nan
    Rrs443 = np.array(f_443['Rrs_443'])
    Rrs443[Rrs443 < 0] = np.nan
    Rrs488 = np.array(f_488['Rrs_488'])
    Rrs488[Rrs488 < 0] = np.nan
    Rrs531 = np.array(f_531['Rrs_531'])
    Rrs531[Rrs531 < 0] = np.nan
    Rrs551 = np.array(f_547['Rrs_547'])
    Rrs551[Rrs551 < 0] = np.nan
    Rrs671 = np.array(f_667['Rrs_667'])
    Rrs671[Rrs671 < 0] = np.nan

    # Raman Corr
    LUT = np.array([[412, 0.003, 0.014, -0.022],
                    [443, 0.004, 0.015, -0.023],
                    [488, 0.011, 0.010, -0.051],
                    [531, 0.015, 0.010, -0.070],
                    [547, 0.0166, 0.010, -0.078],
                    [667, 0.018, 0.010, -0.081]])

    Rrs_ratio = Rrs443 / Rrs551
    Rrs412 = Rrs412 / (1 + (LUT[0][1] * Rrs_ratio + LUT[0][2] * Rrs551 ** LUT[0][3]))
    Rrs443 = Rrs443 / (1 + (LUT[1][1] * Rrs_ratio + LUT[1][2] * Rrs551 ** LUT[1][3]))
    Rrs488 = Rrs488 / (1 + (LUT[2][1] * Rrs_ratio + LUT[2][2] * Rrs551 ** LUT[2][3]))
    Rrs531 = Rrs531 / (1 + (LUT[3][1] * Rrs_ratio + LUT[3][2] * Rrs551 ** LUT[3][3]))
    Rrs551 = Rrs551 / (1 + (LUT[4][1] * Rrs_ratio + LUT[4][2] * Rrs551 ** LUT[4][3]))
    Rrs671 = Rrs671 / (1 + (LUT[5][1] * Rrs_ratio + LUT[5][2] * Rrs551 ** LUT[5][3]))




    (n_row, n_col) = np.shape(Rrs443)
    # print('shape of image :', np.shape(Rrs443))

    # input data
    input1 = Rrs412.flatten()[:, np.newaxis]
    input2 = Rrs443.flatten()[:, np.newaxis]
    input3 = Rrs488.flatten()[:, np.newaxis]
    input4 = Rrs531.flatten()[:, np.newaxis]
    input5 = Rrs551.flatten()[:, np.newaxis]
    input6 = Rrs671.flatten()[:, np.newaxis]
    # input5 = input1 / input3
    # input6 = input2 / input3
    # input7 = input4 / input3

    X = np.concatenate([input1, input2, input3, input4, input5, input6], axis=1)
    # X = np.concatenate([input1, input2, input3, input4, input5, input6, input7], axis=1)
    # 要把X中小于0的改为np.nan， 需要先
    # X = np.concatenate([input1, input2, input3, input4, input5, input6, input7], axis=1).astype('float')
    # X[X < 0] = np.nan

    x_chs = pd.DataFrame(data=X)
    x_chs[x_chs < 0] = np.nan  # delete nan
    # Rrs_raw = np.array(x_chs)
    x_chs['idx'] = list(range(0, np.size(X[:, 0])))    # flag location of each pixel
    x_chs = x_chs.dropna(axis=0, how='any')            # drop nan Rrs

    X_chs = x_chs.copy()
    X_chs.pop('idx')

    # load model
    model_dir = './Model/'
    # with open(model_dir + 'scaler_x.pkl', 'rb') as f:
    #     scaler_x = pickle.load(f)
    model = load_model(model_dir + 'model_Y6.h5', custom_objects={'r2_keras': r2_keras})

    # X_input = scaler_x.transform(X_chs)
    y_predict = model.predict(X_chs)

    #  predict process
    raw = pd.DataFrame(np.hstack([y_predict, x_chs[['idx']]]), columns=['predict', 'idx'])
    predict_1 = pd.DataFrame(data=np.arange(0, np.size(X[:, 1])), columns=['idx'])
    predict = pd.merge(predict_1, raw, how='outer', on='idx')

    output = np.array(predict['predict'])
    output = output.reshape(n_row, n_col)  # reshape output to image size

    # save results, and plot result map with Matlab m_map
    # sio.savemat(r'D:\others\20200829-depth\sat_application\H_classify\result_classify_0525.mat',
    #             mdict={'lon': lon, 'lat': lat, 'h': output})

    output = np.round(output, 2)  # save 3 significant digital

    f_w = nc.Dataset(os.path.join(save_dir, save_name), 'w', format='NETCDF4')

    # define dimensions
    longs = f_w.createDimension('longitude', size=len(lon))
    lats = f_w.createDimension('latitude', size=len(lat))

    # create variables
    lat_w = f_w.createVariable('lat', np.single, 'latitude')
    lon_w = f_w.createVariable('lon', np.single, 'longitude')
    Y_w = f_w.createVariable('Y', np.single, ('latitude', 'longitude'))

    lon_w[:] = lon
    lat_w[:] = lat
    Y_w[:] = output
    f_w.close()


def main():

    data_dir = r'F:/MODIS_MO_Data/Rrs'
    save_dir = r'./sat'

    modis_dates = set([fn[:15] for fn in os.listdir(data_dir)])
    modis_dates = list(modis_dates)
    modis_dates.sort()
    print(modis_dates)

    for i in range(225, len(modis_dates)):
    # for i in range(39, 40):
        save_name = modis_dates[i] + '_Y_py_6b_mar.nc'
        print(save_name)
        print(i)
        model_application(data_dir, modis_dates[i], save_dir, save_name)


if __name__ == '__main__':
    main()
