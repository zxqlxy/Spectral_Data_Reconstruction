import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import random
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


# ======= Some constants =======
delta = 0.01
speed_of_light = 3 * 10 ** 5
DATA_URL = ('./Forward_Model/EIS_270_280.0.spec.det')

element = {'Mg_V': 276.591, 'Mg_VI': 270.391, 'Mg_VII': 278.404, 'Si_VII': 275.354, 'Fe_XIV1': 270.522,
           'Fe_XIV2': 274.182}

### Get wavelength
data = pd.read_csv(DATA_URL)

# This will store all the wave length and will be the same for all
wavelength_str = data.iloc[[0]].values[0][0].split('\t')
wavelength = []
for i in wavelength_str:
    num = i.split('e')
    i = float(num[0]) * 10 ** int(num[1])
    wavelength.append(i)

# This will store all the bins
store_bins = data.iloc[1::2]


def main():
    st.title('Solar Corona')
    st.header('1. Exploration')

    ### Select Different Event
    events = st.sidebar.selectbox("Select Event", ['0','1','2','3','4','5','6','7','8','9','10'])
    if events == '0':
    	event = ''
    elif  events == '1':
    	event = '1'
    elif  events == '2':
    	event = '2'
    elif  events == '3':
    	event = '3'
    elif  events == '4':
    	event = '4'
    elif  events == '5':
    	event = '5'
    elif  events == '6':
    	event = '6'
    elif  events == '7':
    	event = '7'
    elif  events == '8':
    	event = '8'
    elif  events == '9':
    	event = '9'
    elif  events == '10':
    	event = '10'


    URL = './Forward_Model_' + event + '/EIS_270_280.{}.spec.det'
    time = st.slider('time', min_value=0, max_value=5000, value=0, step=5)
    bins = st.slider('bins', min_value=1, max_value=51, step=1)

    filtered_data = load_data_time_bin(event, time, bins)

    st.header('Plot at time: %s bins : %s' % (time, bins))
    plt.plot(wavelength, filtered_data)

    for i in range(len(element)):
        plt.plot([list(element.values())[i], list(element.values())[i]], [0, max(filtered_data)], 'c-.', lw=0.5)
    st.pyplot()
    plt.clf()

    st.header('2. Analysis')

    st.sidebar.title("Solar Corona")

    analysis = st.sidebar.selectbox("Parameters", ['None', 'Amplitude', 'Doppler Shfit', 'FWHM', 'All'])
    if analysis == 'None':
        pass
    elif analysis == 'Amplitude':
        st.subheader('Find Amplitude')
        amplitude = st.slider('amplitude', min_value=0, max_value=5000, value=100, step=5)
        plot_amplitude(data_process(URL, amplitude))
    elif analysis == 'Doppler Shfit':
        st.subheader('Find Doppler Shift')
        doppler_time = st.slider('doppler time', min_value=0, max_value=5000, value=100, step=5)
        plot_doppler_shift(data_process(URL, doppler_time))
    elif analysis == 'FWHM':
        st.subheader('Find FWHM')
        FWHM_time = st.slider('FWHM time', min_value=0, max_value=5000, value=100, step=5)
        plot_FWHM(data_process(URL, FWHM_time))
    elif analysis == 'All':
        st.subheader('Find Amplitude')
        amplitude = st.slider('amplitude', min_value=0, max_value=5000, value=100, step=5)
        plot_amplitude(data_process(URL, amplitude))
        st.subheader('Find Doppler Shift')
        doppler_time = st.slider('doppler time', min_value=0, max_value=5000, value=100, step=5)
        plot_doppler_shift(data_process(URL, doppler_time))
        st.subheader('Find FWHM')
        FWHM_time = st.slider('FWHM time', min_value=0, max_value=5000, value=100, step=5)
        plot_FWHM(data_process(URL, FWHM_time))

    agree = st.checkbox('Plot Correlation')
    if agree:
        plot_corr()

    st.header('3. PCA')

    pca_mode = st.sidebar.selectbox("Choose PCA mode",
                                    ["None","standard", "normal"])
    if pca_mode == "None":
        pass
    elif pca_mode == "standard":
        plot_PCA()
        st.sidebar.success('To continue select "Run the app".')
    elif pca_mode == 'normal':
        plot_PCA('normal')
        st.sidebar.success('To continue select "Run the app".')

    app_mode = st.sidebar.selectbox("Choose training model",
                                    ["None", "kmeans", "RandomForestRegressor"])
    if app_mode == "None":
        pass
    if app_mode == "RandomForestRegressor":
        score = random_regressor()
        st.sidebar.success('To continue select "Run the app".')
    if app_mode == "kmeans":
        score = Kmeans()
        st.sidebar.success('To continue select "Run the app".')


# elif app_mode == "Show the source code":
#     readme_text.empty()
#     st.code(get_file_content_as_string("app.py"))
# elif app_mode == "Run the app":
#     readme_text.empty()
#     run_the_app()


### Inspect the data
@st.cache
def load_data_time_bin(event, i, j):
    """
    This will load the data of certain time and certain bins
    """
    if not event:
    	link = './Forward_Model/EIS_270_280.{}.spec.det'.format(i)
    else:
    	link = './Forward_Model_{}/EIS_270_280.{}.spec.det'.format(event, i)
    # Load time
    data = pd.read_csv(link)
    # Read bins
    y = data.iloc[[2 * j]].values[0][0].split('\t')
    yy = []
    for i in y:
        num = i.split('e')
        i = float(num[0]) * 10 ** int(num[1])
        yy.append(i)
    return yy


# ================   A section where you can input time and bins ================
# # A BUG at number_input
# input_time = st.text_input('time')
# input_bins = st.text_input('bins')

def search_range(lambda_0, y):
    """
    This method is too naive
    This check the maximum of a range of +- 0.1
    """
    ind = []
    for i in range(len(wavelength)):
        if wavelength[i] <= lambda_0 + delta and wavelength[i] >= lambda_0 - delta:
            ind.append(i)

    val = []
    for i in ind:
        val.append(y[i])
    value = max(val)

    return value

def gradient_search(lambda_0, y):
    """
    In this method, we care about range in +- 0.1 which is approximately +- 60 data points.
    """
    for i in range(len(wavelength)):
        if wavelength[i] >= lambda_0:
            start = wavelength[i]
            break

    ind = i

    # Left
    while i > ind - 60:
        save_L = y[i]
        if y[i - 1] >= y[i]:
            save_L = y[i - 1]
            i -= 1
        else:
            break
    # Account for didn't find the maximum
    if i == ind - 60:
        i = ind
        save_L = y[ind]

    # Right
    j = ind
    while j < ind + 60:
        save_R = y[j]
        try:
            if y[j + 1] >= y[j]:
                save_R = y[j + 1]
                j += 1
            else:
                break
        except IndexError:
            j = ind
            save_R = y[ind]
            break
    # Account for didn't find the maximum
    if j == ind + 60:
        j = ind
        save_R = y[ind]

    return (save_L, i) if save_L > save_R else (save_R, j)

def doppler_shift(lambda_, ele, wavelength):
    """
    This is in the unit of km/s
    """
    return speed_of_light * abs(element[ele] - wavelength[lambda_]) / element[ele]

def find_FWHM(half_max, lambda_, y):
    # Left
    i = lambda_
    while y[i] >= half_max:
        i -= 1

    # Right
    j = lambda_
    while y[j] >= half_max:
        j += 1

    return wavelength[j] - wavelength[i]


@st.cache
def data_process(URL, i):
    data = pd.read_csv(URL.format(i))

    bins = data.iloc[1::2]
    entry = data.iloc[2::2]

    # Initialize the matrix
    res = np.empty([len(entry.values), len(element)], dtype=list)
    for i in range(len(entry.values)):
        datt = entry.values[i][0].split('\t')
        dat = []

        # Process str to number
        for j in datt:
            num = j.split('e')
            j = float(num[0]) * 10 ** int(num[1])
            dat.append(j)

        num = 0
        # Loop over all the elements
        for ele in element:

            col = []
            # amp_1 = search_range(element[ele], dat)
            amp_2, lambda_ = gradient_search(element[ele], dat)
            # This way of computing doppler shift seems to be slow
            # dop = doppler_shift(lambda_, ele, wavelength)

            dop = speed_of_light * (wavelength[lambda_] - element[ele]) / element[ele]

            try:
                FWHM = find_FWHM(amp_2 / 2, lambda_, dat)
            except IndexError:
                FWHM = 0

                #  Error seems to be caused by extremly small amplitude
                # print('jhhh')
                # print(amp_2)
                # print(lambda_)
                # print(dat[lambda_])

            # bins, element, amplitude, FWHM
            col = [bins.values[i][0].split('e'), ele, amp_2, dop, FWHM]
            res[i][num] = col
            num += 1

    return res


# ================ Some interpretation on bins ================
# By the structure of the detector, bins on the side (around 0 or 50) are cooler temperature (which will create Mg)
# and bins in the middle are hotter temperature (which will create Fe).

def plot_amplitude(res):
    x = store_bins.values.flatten()
    x = [float(i) for i in x]

    for j in range(len(element)):
        y = res[:, j]
        yy = []
        for i in y:
            yy.append(i[2])

        plt.subplot(2, 3, j + 1)
        plt.plot(x, yy)
        plt.title(list(element)[j])
        plt.xticks(np.arange(0, 51, 10))
        plt.axis()
    plt.suptitle('Plot of Amplitude', y=1)
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.10, right=0.95, hspace=0.4,
                        wspace=0.35)
    st.pyplot()
    plt.clf()


def plot_doppler_shift(res):
    x = store_bins.values.flatten()
    x = [float(i) for i in x]

    for j in range(len(element)):
        y = res[:, j]
        yy = []
        for i in y:
            yy.append(i[3])

        plt.subplot(2, 3, j + 1)
        plt.plot(x, yy)
        plt.title(list(element)[j])
        plt.xticks(np.arange(0, 51, 10))

        plt.axis()
    plt.suptitle('Plot of dopper shift', y=1)
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.10, right=0.95, hspace=0.4,
                        wspace=0.35)
    st.pyplot()
    plt.clf()


def plot_FWHM(res):
    x = store_bins.values.flatten()
    x = [float(i) for i in x]

    for j in range(len(element)):
        y = res[:, j]
        yy = []
        for i in y:
            yy.append(i[4])

        plt.subplot(2, 3, j + 1)
        plt.plot(x, yy)
        plt.title(list(element)[j])
        plt.xticks(np.arange(0, 51, 10))
        plt.axis()
    plt.suptitle('Plot of FWHM', y=1)
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.10, right=0.95, hspace=0.4,
                        wspace=0.35)
    st.pyplot()
    plt.clf()


# iris = datasets.load_iris()

# print(iris.data)
# print(iris.target)


@st.cache
def process_all(start, end):
    """
    This will process all the data from 0 to num
    """
    processed = []
    for time in range(start, end, 5):
        data = data_process(time)
        for i in range(len(data)):
            num = 1
            for j in data[i]:
                vector = [time, i, num]
                vector.extend(j[2:])
                processed.append(vector)
                num += 1

    return processed


@st.cache
def process_all_nan(start, end, prob):
    """
    This will process all the data from 0 to num, and the probability of retaining the data
    """
    processed = []
    nan = []
    for time in range(start, end, 5):
        data = data_process(time)
        for i in range(len(data)):
            num = 1
            for j in data[i]:
                vector = [time, i, num]
                vector.extend(j[2:])

                rand = random.random()
                if rand < prob:
                    processed.append(vector)
                else:
                    processed.append([time, i, num, np.nan, np.nan, np.nan])
                    nan.append([time, i, num, np.nan, np.nan, np.nan])
                num += 1

    return processed, nan


def plot_PCA(option='standard'):
    pca = decomposition.PCA()
    processed = process_all(0, 2000)
    if option == 'standard':
        standard = StandardScaler()
        standard.fit(processed)  # Standardize the data
        stand = standard
        processed = standard.transform(processed)
        pca.fit(processed)
        newdata = pca.transform(processed)
    elif option == 'normal':
        pca.fit(processed)
        newdata = pca.transform(processed)
    # print(pca.get_covariance())
    plt.matshow(pca.components_, cmap='viridis')
    # plt.yticks([0,1,2],['1st Comp','2nd Comp','3rd Comp'],fontsize=10)
    plt.colorbar()
    plt.xticks(range(6), ['time', 'bins', 'element', 'Amplitude', 'Doppler', 'FWHM'], rotation=65, ha='left')
    plt.tight_layout()
    st.pyplot()
    plt.clf()

    # print(principalComponents)


def plot_corr():
    data = process_all(0, 2000)
    df = pd.DataFrame(data, columns=['time', 'bins', 'element', 'Amplitude', 'Doppler', 'FWHM'])
    corr = df.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(df.columns), 1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    # ax.set_yticks(ticks)
    # ax.set_xticklabels(data.columns)
    # ax.set_yticklabels(data.columns)
    st.pyplot()


def random_regressor():
    """
    Multiple Regressor + random forest seems to be able to create something but not accurate enough
    :return:
    """
    processed_nan, nan = process_all_nan(0, 500, 0.9)
    # All data
    df = pd.DataFrame(processed_nan)
    notnans = df.notnull().all(axis=1)
    df_notnans = df[notnans]

    # Nan data
    df_nan = pd.DataFrame(nan)

    X_train, X_test, y_train, y_test = train_test_split(df_notnans[[0, 1, 2]], df_notnans[[3, 4, 5]],
                                                        train_size=0.75,
                                                        random_state=4)
    regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,
                                                              max_depth=20,
                                                              random_state=0))

    # Fit on the train data
    regr_multirf.fit(X_train, y_train)

    # Check the prediction score
    score = regr_multirf.score(X_test, y_test)
    print("The prediction score on the test data is {:.2f}%".format(score * 100))

    processed = process_all(0, 500)
    data = pd.DataFrame(processed, columns=['time', 'bins', 'element', 'Amplitude', 'Doppler', 'FWHM'])

    predicted = regr_multirf.predict(df_nan[[0, 1, 2]])
    pred_df = pd.DataFrame(predicted, columns=['Amplitude', 'Doppler', 'FWHM'])

    original = []
    for row in df_nan.iterrows():
        item = data.loc[lambda df: df['time'] == row[1][0]].loc[lambda df: df['bins'] == row[1][1]].loc[
            lambda df: df['element'] == row[1][2]]
        original.append(item)
    original_df = pd.DataFrame(original)

    st.pyplot()
    plt.clf()
    plt.plot(range(len(df_nan)), pred_df[[0]])
    st.pyplot()

    return score


@st.cache
def Kmeans():
    """
    Kmeans seems to be unable to find anything useful.
    :return:
    """
    X = process_all(0, 500)
    X = np.array([i[3:] for i in X])
    min1 = np.min(X[:, 0])
    min2 = np.min(X[:, 1])
    min3 = np.min(X[:, 2])
    a = np.log(X[:, 0] - min1 + 1)
    b = np.log(X[:, 1] - min2 + 1)
    c = np.log(X[:, 2] - min3 + 1)
    kmeans = KMeans(n_clusters=6, random_state=0).fit(X)
    print(kmeans.labels_)
    estimators = [('k_means__8', KMeans(n_clusters=8)),
                  ('k_means__6', KMeans(n_clusters=6)),
                  ('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1,
                                                   init='random'))]

    fignum = 1
    titles = ['8 clusters', '6 clusters', '3 clusters, bad initialization']
    for name, est in estimators:
        fig = plt.figure(fignum, figsize=(4, 3))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        est.fit(X)
        labels = est.labels_

        ax.scatter(a, b, c,
                   c=labels.astype(np.float), edgecolor='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Doppler Shift')
        ax.set_zlabel('FWHM')
        ax.set_title(titles[fignum - 1])
        ax.dist = 12

        # fig.show()
        plt.show()
        st.pyplot()
        # plt.clf()
        fignum = fignum + 1
    pass


# processed = []
# for time in range(0,2000, 5):
# 	data = data_process(time)
# 	for i in range(len(data)):
# 		num = 1
# 		for j in data[i]:
# 			vector = [time, i, num]
# 			vector.append(j[2])

# 			rand = random.random()
# 			if rand < 0.9:
# 				processed.append(vector)
# 			else:
# 				processed.append([time, i, num, np.nan])
# 			num += 1


# # def random_regressor():
# # processed_nan = process_all_nan(2000, 0.9)
# df = pd.DataFrame(processed)
# notnans = df.notnull().all(axis=1)
# df_notnans = df[notnans]
# # print(notnans)
# X_train, X_test, y_train, y_test = train_test_split(df_notnans[[0,1,2]], df_notnans[[3]],
#                                                     train_size=0.75,
#                                                     random_state=4)
# regr_multirf = RandomForestRegressor(n_estimators = 100, 
# 													max_depth=20,
#                                                     random_state=0)

# # Fit on the train data
# regr_multirf.fit(X_train, y_train)

# # Check the prediction score
# score = regr_multirf.score(X_test, y_test)
# print("The prediction score on the test data is {:.2f}%".format(score*100))


# processed = process_all(2000)
# data = pd.DataFrame(processed) 


# predicted = regr_multirf.predict(data[[0,1,2]])
# pred_df = pd.DataFrame(predicted)
# print(predicted)
# plt.plot(range(len(predicted[[0]])), predicted[[0]])
# st.pyplot()
# plt.clf()
# plt.plot(range(1000), data[[3]][:1000])
# st.pyplot()


# return score

# 

# from mpl_toolkits.mplot3d import Axes3D

# @st.cache()
# def proc():
# 	processed = []
# 	for time in range(0,500, 5):
# 		data = data_process(time)
# 		for i in range(len(data)):
# 			num = 1
# 			for j in data[i]:
# 				vector = [time, i, num]
# 				vector.append(j[2])
# 				processed.append(vector)
# 				num += 1
# 	return processed

# process = proc()
# dff = pd.DataFrame(process)
# processed = StandardScaler().fit_transform(dff[[0,1,2]])
# print(dff)
# pca = decomposition.PCA(n_components = 2)
# df = pd.DataFrame(processed)
# principalComponents = pca.fit_transform(df[[0,1,2]])
# final = pd.DataFrame(principalComponents, columns = ['principal component 1', 'principal component 2'])
# print(final)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# i = 0
# for i in range(len(final)):
# 	ax.scatter(final.iloc[i, 0], final.iloc[i, 1],dff.iloc[i, 3])
# 	i += 1
# 	if i > 1000:
# 		break

# st.pyplot()


if __name__ == '__main__':
    main()
