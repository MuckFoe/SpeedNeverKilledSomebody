import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import mode
from argparse import ArgumentParser

# source: https://medium.com/@ggaighernt/optical-flow-and-motion-detection-5154c6ba4419
# more: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
# even more: https://nanonets.com/blog/optical-flow/

# this is dense optical flow

if __name__ == '__main__':
    ap = ArgumentParser() # arguments to do different stuff
    ap.add_argument('-rec', '--record', default=False, action='store_true', help='Record?')
    ap.add_argument('-pscale', '--pyr_scale', default=0.5, type=float,
                    help='Image scale (<1) to build pyramids for each image')
    ap.add_argument('-l', '--levels', default=3, type=int, help='Number of pyramid layers')
    ap.add_argument('-w', '--winsize', default=15, type=int, help='Averaging window size')
    ap.add_argument('-i', '--iterations', default=3, type=int,
                    help='Number of iterations the algorithm does at each pyramid level')
    ap.add_argument('-pn', '--poly_n', default=5, type=int,
                    help='Size of the pixel neighborhood used to find polynomial expansion in each pixel')
    ap.add_argument('-psigma', '--poly_sigma', default=1.1, type=float,
                    help='Standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion')
    ap.add_argument('-th', '--threshold', default=10.0, type=float, help='Threshold value for magnitude')
    ap.add_argument('-p', '--plot', default=False, action='store_true', help='Plot accumulators?')
    ap.add_argument('-rgb', '--rgb', default=False, action='store_true', help='Show RGB mask?')
    ap.add_argument('-s', '--size', default=10, type=int, help='Size of accumulator for directions map')

    args = vars(ap.parse_args()) #retrieve args

    directions_map = np.zeros([args['size'], 5]) # initialize directions map

    cap = cv.VideoCapture(0) # get videoinput
    if args['record']: # record video
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        codec = cv.VideoWriter_fourcc(*'MPEG')
        out = cv.VideoWriter('out.avi', codec, 10.0, (w, h))

    if args['plot']:
        plt.ion()

    frame_previous = cap.read()[1]
    gray_previous = cv.cvtColor(frame_previous, cv.COLOR_BGR2GRAY) # optical flow calculation see opticalflowalgorithm.pdf
    hsv = np.zeros_like(frame_previous) # H - direction, S - saturation, V - magnitude
    hsv[:, :, 1] = 255 # max
    param = {
        'pyr_scale': args['pyr_scale'],
        'levels': args['levels'],
        'winsize': args['winsize'],
        'iterations': args['iterations'],
        'poly_n': args['poly_n'],
        'poly_sigma': args['poly_sigma'],
        'flags': cv.OPTFLOW_LK_GET_MIN_EIGENVALS
    }

    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(gray_previous, gray, None, **param) # this function returns a ndarray with the same number of rows and columns that both images in the calculation, but with two channels: the first one with the x, the second with the y coordinates.
        mag, ang = cv.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=True) #get magnitude and direction (in degrees not radians!) through cartToPolar function from the x and y coordninates
        ang_180 = ang / 2 # H channel (see hsv above) has value from 0-179, so need to halve for direction
        gray_previous = gray # this frame stored for the next calulation

        move_sense = ang[mag > args['threshold']] # filter direction with magnitude higher than threshold (also filter noises)
        move_mode = mode(move_sense)[0] # calculate mode, see wich is predominant in the image

        if 10 < move_mode <= 100:  # accumulator stuff transfers movement to 2d directions , angles + 10 percent because of inaccurances
            directions_map[-1, 0] = 1
            directions_map[-1, 1:] = 0
            directions_map = np.roll(directions_map, -1, axis=0)
        elif 100 < move_mode <= 190:
            directions_map[-1, 1] = 1
            directions_map[-1, :1] = 0
            directions_map[-1, 2:] = 0
            directions_map = np.roll(directions_map, -1, axis=0)
        elif 190 < move_mode <= 280:
            directions_map[-1, 2] = 1
            directions_map[-1, :2] = 0
            directions_map[-1, 3:] = 0
            directions_map = np.roll(directions_map, -1, axis=0)
        elif 280 < move_mode or move_mode < 10:
            directions_map[-1, 3] = 1
            directions_map[-1, :3] = 0
            directions_map[-1, 4:] = 0
            directions_map = np.roll(directions_map, -1, axis=0)
        else:
            directions_map[-1, -1] = 1
            directions_map[-1, :-1] = 0
            directions_map = np.roll(directions_map, 1, axis=0)

        if args['plot']:
            plt.clf()
            plt.plot(directions_map[:, 0], label='Down')
            plt.plot(directions_map[:, 1], label='Right')
            plt.plot(directions_map[:, 2], label='Up')
            plt.plot(directions_map[:, 3], label='Left')
            plt.plot(directions_map[:, 4], label='Waiting')
            plt.legend(loc=2)
            plt.pause(1e-5)
            plt.show()

        if move_mode.size != 0:
            text = repr(move_mode[0])
        else:
            text = ''

        hsv[:, :, 0] = ang_180 # direction from above
        hsv[:, :, 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX) # normalize values to 255 to see little magnitudes
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR) #convert HSV to RGB

        frame = cv.flip(frame, 1)
        cv.putText(frame, text, (30, 90), cv.FONT_HERSHEY_COMPLEX, frame.shape[1] / 500, (0, 0, 255), 2)

        k = cv.waitKey(1) & 0xff
        if k == ord('q'):
            break
        if args['record']:
            out.write(frame)
        if args['rgb']:
            cv.imshow('Mask', rgb)
        cv.imshow('Frame', frame)
        k = cv.waitKey(1) & 0xff
        if k == ord('q'):
            break

    cap.release()
    if args['record']:
        out.release()
    if args['plot']:
        plt.ioff()
    cv.destroyAllWindows()