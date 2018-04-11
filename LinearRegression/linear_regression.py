import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import sklearn
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DeepLearningIntro\LinearRegression')

challenge_dataset_path = "./challenge_dataset.txt"
global_co2_path = "./global_co2.csv"
annual_temp_path = "./annual_temp.csv"
without_sklearn_data_path = "./without_sklearn_dataset.csv"

show_plots = True


def on_challenge_data():
    # read the file into dataframe
    print("Showing results for challenge dataset")
    df = pd.read_csv(challenge_dataset_path, names=['X','Y'])
    if show_plots:
        sns.regplot(x='X', y='Y', data=df, fit_reg=False)
        plt.show()

    # split the data into train and test
    X_train, X_test, y_train, y_test = np.asarray(train_test_split(df['X'], df['Y'], test_size=0.1))

    reg = LinearRegression()
    reg.fit(X_train.values.reshape(-1,1), y_train.values.reshape(-1,1))
    print('Score: ', reg.score(X_test.values.reshape(-1,1), y_test.values.reshape(-1,1)))

    x_line = np.arange(5,25).reshape(-1,1)
    if show_plots:
        sns.regplot(x=df['X'], y=df['Y'], data=df, fit_reg=False)
        plt.plot(x_line, reg.predict(x_line))
        plt.show()

    return 0


def on_global_data():
    # read the file into dataframe
    print("Showing results for global Temp/CO2 dataset")
    co2_df = pd.read_csv(global_co2_path)
    temp_df = pd.read_csv(annual_temp_path)

    # Clean data
    # Keep only coloumn Year, Total
    co2_df = co2_df[['Year', 'Total']]
    # Keep data for year 1960 - 2010
    co2_df = co2_df[co2_df['Year'] >= 1960]
    # Rename and reset columns index
    co2_df.columns = ['Year','CO2']
    co2_df.reset_index(drop=True, inplace=True)

    # Keep only one source
    temp_df = temp_df[temp_df.Source != 'GISTEMP']
    # Drop name of source
    temp_df.drop(['Source'], inplace=True, axis=1)
    # re-arrange items in reverse order
    temp_df = temp_df.reindex(index=temp_df.index[::-1])
    # Keep data for year 1960 - 2010
    temp_df = temp_df[(temp_df.Year >= 1960) & (temp_df.Year <= 2010)]
    # Rename and reset columns index
    temp_df.columns = ['Year','Temperature']
    temp_df.reset_index(drop=True, inplace=True)

    # concatenate
    climate_change_df = pd.concat([co2_df, temp_df.Temperature], axis=1)

    # 3D projection of data
    if show_plots:
        fig = plt.figure()
        # fig.set_size_inches(12.5, 7.5)
        fig.add_subplot(111, projection='3d')
        ax = Axes3D(fig)
        ax.scatter(xs=climate_change_df['Year'], ys=climate_change_df['Temperature'], zs=climate_change_df['CO2'])
        ax.set_ylabel('Relative tempature'); ax.set_xlabel('Year'); ax.set_zlabel('CO2 Emissions')
        ax.view_init(10, -45)
        plt.show()

    # 2D projection of data
    if show_plots:
        f, axarr = plt.subplots(2, sharex=True)
        # f.set_size_inches(12.5, 7.5)
        axarr[0].plot(climate_change_df['Year'], climate_change_df['CO2'])
        axarr[0].set_ylabel('CO2 Emissions')
        axarr[1].plot(climate_change_df['Year'], climate_change_df['Temperature'])
        axarr[1].set_xlabel('Year')
        axarr[1].set_ylabel('Relative temperature')
        plt.show()

    # Linear Regression
    X = climate_change_df.as_matrix(['Year'])
    Y = climate_change_df.as_matrix(['CO2', 'Temperature']).astype('float32')
    # split the data into train and test
    X_train, X_test, y_train, y_test = np.asarray(train_test_split(X, Y, test_size=0.1))

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    print('Score: ', reg.score(X_test.reshape(-1, 1), y_test))

    x_line = np.arange(1960,2011).reshape(-1,1)
    p = reg.predict(x_line).T

    # 3D projection of data with regression
    if show_plots:
        fig = plt.figure()
        # fig.set_size_inches(12.5, 7.5)
        fig.add_subplot(111, projection='3d')
        ax = Axes3D(fig)
        ax.scatter(xs=climate_change_df['Year'], ys=climate_change_df['Temperature'], zs=climate_change_df['CO2'])
        ax.set_ylabel('Relative tempature'); ax.set_xlabel('Year'); ax.set_zlabel('CO2 Emissions')
        ax.plot(xs=x_line, ys=p[1], zs=p[0], color='green')
        ax.view_init(10, -45)
        plt.show()

    # 2D projection of data with regression
    if show_plots:
        f, axarr = plt.subplots(2, sharex=True)
        # f.set_size_inches(12.5, 7.5)
        axarr[0].plot(climate_change_df['Year'], climate_change_df['CO2'])
        axarr[0].plot(x_line, p[0])
        axarr[0].set_ylabel('CO2 Emissions')
        axarr[1].plot(climate_change_df['Year'], climate_change_df['Temperature'])
        axarr[1].plot(x_line, p[1])
        axarr[1].set_xlabel('Year')
        axarr[1].set_ylabel('Relative temperature')
        plt.show()

    return 0


def UsingSklearn():

    on_challenge_data()
    on_global_data()

    return 0


# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    # b, m, points = initial_b, initial_m, points
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))


def step_gradient(b_current, m_current, points, learningRate):
    # b_current, m_current, points, learningRate = b, m, np.array(points), learning_rate
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return new_b, new_m


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    # points, starting_b, starting_m, learning_rate, num_iterations = points, initial_b, initial_m, learning_rate, num_iterations
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, points, learning_rate)
    return b, m


def predict_line(b, m, x):

    return (m * x) + b;


def withoutSklearn():
    """
    The dataset is collection of test scores and amount of hours of study.
    The left coloumn is amount of hours of study(X values), the right coloumn are the test scores(Y values).
    """
    print("Linear Regression without Sklearn")
    df_points = pd.read_csv(without_sklearn_data_path, header = None)
    df_points.columns = ['X','Y']

    if show_plots:
        sns.regplot(x='X', y='Y', data=df_points, fit_reg=False)
        plt.show()

    points = np.array(df_points)
    learning_rate = 0.0001
    # initial y-intercept guess
    initial_b = 0
    # initial slope guess
    initial_m = 0
    # with more iteration value gets better
    num_iterations = 1000
    compute_error = compute_error_for_line_given_points(initial_b, initial_m, points)

    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error))
    print("Running...")
    b, m = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    compute_error = compute_error_for_line_given_points(b, m, points)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error))

    x_line = np.arange(30,70).reshape(-1,1)
    if show_plots:
        sns.regplot(x='X', y='Y', data=df_points, fit_reg=False)
        plt.plot(x_line, predict_line(b, m, x_line))
        plt.show()

    return 0


def main():

    UsingSklearn()
    withoutSklearn()

    return 0


if __name__ == '__main__':
    main()
