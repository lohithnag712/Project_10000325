from __future__ import print_function

from os import listdir
from os.path import isfile, join, exists
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as py
from tqdm import tqdm
import math

import warnings
warnings.filterwarnings('ignore')

from movement import utils
import movement.config as CONFIG


def correct_shots(game_shots, movement, events):

    fixed_shots = pd.DataFrame(columns=game_shots.columns)
    fixed_shots['QUARTER'] = 0

    # count = 0
    for ind, shot in game_shots.iterrows():

        if shot['SHOT_DISTANCE'] > 4:
            try:
                event_id = shot['GAME_EVENT_ID']
                loc_x = shot['LOC_X']
                loc_y = shot['LOC_Y']

                movement_around_shot = movement.loc[movement.event_id.isin([event_id, event_id - 1])]
                movement_around_shot.drop_duplicates(subset=['game_clock','quarter'], inplace=True)
                sec_before, sec_after = 4, 3
                mask = (movement_around_shot['game_clock'] > shot['EVENTTIME']-sec_after) & (
                    movement_around_shot['game_clock'] < shot['EVENTTIME']+sec_before)
                movement_around_shot = movement_around_shot[mask]

                game_clock_time = movement_around_shot.loc[movement_around_shot.team_id == -1, 'game_clock'].values
                ball_height = movement_around_shot.loc[movement_around_shot.team_id == -1, 'radius'].values

                ball_x = movement_around_shot.loc[movement_around_shot.team_id == -1, 'x_loc'].values
                ball_y = movement_around_shot.loc[movement_around_shot.team_id == -1, 'y_loc'].values

                # find closest distance to nba marked shot location within +/- 1-2 sec of closest by y dimension
                ind_y = np.argmin(np.abs(loc_y-ball_y))
    #             print(ind_y, (loc_x, loc_y), ball_x[np.max((0,ind_y-50)):25], ball_y[np.max((0,ind_y-50)):25])
                shot_ind = np.argmin(np.abs(loc_x-ball_x[np.max((0,ind_y-50)):ind_y+25]))
                shot_time = game_clock_time[shot_ind]
                peak_ind = shot_ind + np.argmax(ball_height[shot_ind:(shot_ind+50)])
                rim_gap = np.argmin(np.abs(10 - ball_height[peak_ind:(peak_ind+25)]))
                rim_ind = peak_ind + rim_gap
                peak_reb_gap = np.argmax(ball_height[rim_ind:(rim_ind+50)])
                peak_reb_ind = peak_ind + rim_gap + peak_reb_gap
                shot['reb_height'] = ball_height[peak_reb_ind]
    #             print(shot_time, peak_ind-shot_ind, rim_gap, peak_reb_gap, shot['reb_height'])

                # Rebound (opportunity) location is 1st coordinates where ball drops below 8 feet
                ind_gap = next(x[0] for x in enumerate(ball_height[peak_reb_ind:(peak_reb_ind+50)]) if x[1] < 8)

                # Or whenever ball's flight is interrupted, not a smooth curve anymore
                size = 2
                order = 3
                params = (game_clock_time[(peak_reb_ind+1):(peak_reb_ind+50)], ball_height[(peak_reb_ind+1):(peak_reb_ind+50)], size, order)
                velocity_smoothed = smooth(*params, deriv=1)
    #             position_delta = [ball_height[ind+1]-ball_height[ind] for ind in range(peak_reb_ind,(peak_reb_ind+50))]
                ind_gap2 = next(x[0] for x in enumerate(velocity_smoothed) if x[1] > 0)
                reb_ind = (peak_reb_ind+1) + (ind_gap if ind_gap < ind_gap2 else ind_gap2)
                # print(game_clock_time[peak_reb_ind], ind_gap, ind_gap2, velocity_smoothed)
                reb_time = game_clock_time[reb_ind]
                reb_rho, reb_angle = cart2pol((ball_x[reb_ind], ball_y[reb_ind]))
                shot['reb_time'] = reb_time
                shot['reb_rho'] = reb_rho
                shot['reb_angle'] = reb_angle

                # doesn't work well given the sometimes noisy data
    #             shot_window = acceleration_smoothed[np.max([0, max_ind - 25]): max_ind]
    #             shot_min_ind = np.argmin(shot_window)
    #             shot_ind = max_ind - shot_min_ind
    #             shot_time = game_clock_time[shot_ind]

                quarter = movement_around_shot.loc[:, 'quarter'].values[0]
                movement_around_shot = movement_around_shot.loc[movement_around_shot.game_clock == shot_time, :]

                shot['QUARTER'] = quarter
                shot['shot_time'] = shot_time
                shot['x'] = movement_around_shot.loc[movement_around_shot.team_id == -1, 'x_loc'].values[0]
                shot['y'] = movement_around_shot.loc[movement_around_shot.team_id == -1, 'y_loc'].values[0]

                # Using NBA provided shot X and Y location instead of calculated coordinates
                rho, phi = cart2pol((shot['LOC_X'], shot['LOC_Y']))
                shot['shot_rho'] = rho
                shot['shot_angle'] = phi

                # count += 1
                # size = 10
                # order = 3
                # if count > 7 : return fixed_shots
                # if count < 7 :
                #     print(shot[main_features])
                #     params = (game_clock_time, ball_height, size, order)
                #     position_smoothed = smooth(*params, deriv=0)
                #     velocity_smoothed = smooth(*params, deriv=1)
                #     acceleration_smoothed = smooth(*params, deriv=2)
                #     max_ind = np.argmax(position_smoothed)
                #     plots = [
                #         ["Position", ball_height],
                #         ["Position Smoothed", position_smoothed],
                #         ["Velocity", velocity_smoothed],
                #         ["Acceleration", acceleration_smoothed]
                #     ]
                #     create_figure(order, shot_time)
                #     plot(game_clock_time, plots, shot_ind)
                #
                #     for dimension in ball_x, ball_y :
                #         params = (game_clock_time, dimension, size, order)
                #         position_smoothed = smooth(*params, deriv=0)
                #         velocity_smoothed = smooth(*params, deriv=1)
                #         acceleration_smoothed = smooth(*params, deriv=2)
                #         max_ind = np.argmax(position_smoothed)
                #         plots = [
                #             ["Position", dimension],
                #             ["Position Smoothed", position_smoothed],
                #             ["Velocity", velocity_smoothed],
                #             ["Acceleration", acceleration_smoothed]
                #         ]
                #         create_figure(order, shot_time)
                #         plot(game_clock_time, plots, shot_ind)

            except Exception as err:
                continue

        fixed_shots = fixed_shots.append(shot)

    return fixed_shots

# Polar Conversions
###########################################
def cart2pol(row):
    x = row[0]
    y = row[1]
    rho = round(np.sqrt(x**2 + y**2)/10, 2)
    # not standard polar, orienting to degrees from axis intersecting both baskets
    radian_to_degrees = 57.2958  # convert to degrees
    phi = radian_to_degrees * (3.1416/2 + np.arctan2(y, x) )
    phi = round(phi % 360, 2)
    row = (rho,phi)
    return row

def pol2cart(row):
    rho = row[0]
    phi = row[1]
    x = rho * np.cos(3.1416 - phi)
    y = rho * np.sin(3.1416 - phi)
    row = [x,y]
    return row


def load_shots():
    """
    Load shots only from game events
    """
    shots = pd.read_csv('%s/%s' % (CONFIG.data.shots.dir, 'shots.csv'))
    shots.loc[:, 'EVENTTIME'] = utils.convert_time(minutes=shots.loc[:, 'MINUTES_REMAINING'].values, seconds = shots.loc[:, 'SECONDS_REMAINING'].values)
    shots['GAME_ID'] = '00' + shots['GAME_ID'].astype(int).astype(str)

    return shots


def sg_filter(x, m, k=0):
    """
    x = Vector of sample times
    m = Order of the smoothing polynomial
    k = Which derivative
    """
    mid = int(len(x) / 2)
    a = x - x[mid]
    expa = lambda x: list(map(lambda i: i**x, a))
    A = np.r_[list(map(expa, range(0,m+1)))].transpose()
    Ai = np.linalg.pinv(A)

    return Ai[k]


def smooth(x, y, size=5, order=2, deriv=0):

    if deriv > order:
        raise Exception("deriv must be <= order")

    n = len(x)
    m = size

    result = np.zeros(n)

    for i in range(m, n-m):
        start, end = i - m, i + m + 1
        f = sg_filter(x[start:end], order, deriv)
        result[i] = np.dot(f, y[start:end])

    if deriv > 1:
        result *= math.factorial(deriv)

    return result

def plot(t, plots, shot_ind):
    n = len(plots)

    for i in range(0,n):
        label, data = plots[i]

        plt = py.subplot(n, 1, i+1)
        plt.tick_params(labelsize=8)
        py.grid()
        py.xlim([t[0], t[-1]])
        py.ylabel(label)

        py.plot(t, data, 'k-')
        py.scatter(t[shot_ind], data[shot_ind], marker='*', c='g')

    py.xlabel("Time")
    py.show()
    py.close()

def create_figure(order, shot_time):
    fig = py.figure(figsize=(8,6))
    nth = 'th'
    if order < 4:
        nth = ['st','nd','rd','th'][order-1]

    title = "Shot Time: %s" % (shot_time)

    fig.text(.5, .92, title, horizontalalignment='center')


if __name__ == '__main__':
    game_dir = CONFIG.data.movement.converted.dir
    # game_dir = 'C:/Users/Jason/Documents/Capstone/stage1/data/test_converted'
    event_dir = CONFIG.data.events.dir

    features = ['LOC_X','LOC_Y','ACTION_TYPE','SHOT_MADE_FLAG','SHOT_DISTANCE','PLAYER_ID','PLAYER_NAME',
            'SHOT_TYPE','SHOT_ZONE_AREA','SHOT_ZONE_BASIC','SHOT_ZONE_RANGE','TEAM_NAME','TEAM_ID',
            'EVENTTIME','GAME_EVENT_ID','GAME_DATE','HTM','VTM','GAME_ID']
    main_features = ['shot_rho','shot_angle','reb_rho','reb_angle','reb_height','reb_time','x','y']

    # names = []
    # for f in listdir(game_dir):
    #     if isfile(join(game_dir, f)):
    #         m = re.match(r'\d+', f)
    #         names.append(m.string[m.start() : m.end()])
    # if len(names) > 0:
    #     games = names

    games = utils.get_games()
    events = utils.get_events(event_dir, games)
    shots = load_shots()
    fixed_shots = pd.DataFrame(columns=features)

    count = 0
    for game in tqdm(games):
        # if count < 1 :
        #     count += 1
        try:
            game_movement = pd.read_csv('%s/%s_converted.csv' % (game_dir, game), compression='gzip')
            game_shots = shots.loc[shots.GAME_ID == game, :]
            game_events = events.loc[events.GAME_ID == game, :]
        except IOError as err:
            print(err)
            continue

        fixed_shots = fixed_shots.append(correct_shots(game_shots, game_movement, game_events))

    fixed_shots.to_csv('%s/%s_test_final.csv' % (CONFIG.data.shots.dir, 'shots'), index=False, compression='gzip')