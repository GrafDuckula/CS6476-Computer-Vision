"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2
import random

random.seed(30)

# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.array([init_x, init_y, 0., 0.])  # state
        self.sig = np.eye(4)
        self.sigDt = Q
        self.sigMt = R
        self.Dt = np.eye(4)
        self.Dt[0, 2] = 1
        self.Dt[1, 3] = 1
        self.Mt = np.zeros((2, 4))
        self.Mt[0, 0] = 1
        self.Mt[1, 1] = 1


    def predict(self):
        self.state = self.Dt.dot(self.state)
        self.sig = self.Dt.dot(self.sig.dot(self.Dt.T) + self.sigDt)

    def correct(self, meas_x, meas_y):
        Yt = np.array([meas_x, meas_y]).T
        Kt = self.sig.dot(self.Mt.T).dot(np.linalg.inv(self.Mt.dot(self.sig).dot(self.Mt.T) + self.sigMt))
        self.state = self.state + Kt.dot(Yt - self.Mt.dot(self.state))
        self.sig = (np.eye(4) - Kt.dot(self.Mt)).dot(self.sig)

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        #
        self.template = template
        self.frame = frame

        self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        self.frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        num_row = self.frame.shape[0]
        num_col = self.frame.shape[1]

        self.best_scale = 1.0

        # self.particles = np.c_[np.random.randint(0,num_col,self.num_particles), np.random.randint(0,num_row,self.num_particles)]  # Initialize your particles array. Read the docstring.

        self.particles = np.c_[np.random.randint(int(self.template_rect['x']), int(self.template_rect['x'] + self.template_rect['w']), self.num_particles),
                               np.random.randint(int(self.template_rect['y']), int(self.template_rect['y'] + self.template_rect['h']), self.num_particles)]  # Initialize your particles array. Read the docstring.


        # print(self.particles)

        self.weights = np.full((self.num_particles, 1), 1.0/self.num_particles)  # Initialize your weights array. Read the docstring.
        # print (self.weights.shape)

        # Initialize any other components you may need when designing your filter.


    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        m = template.shape[0]
        n = template.shape[1]
        fm = frame_cutout.shape[0]
        fn = frame_cutout.shape[1]

        # print(template)
        # print(frame_cutout)

        if m == fm and n == fn:
            # print (np.sum(np.square(template - frame_cutout))*1.0/m/n)
            return np.sum(np.square(template.astype('float64') - frame_cutout.astype('float64')))*1.0/m/n
        else:
            return np.inf


    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """
        # prob = self.weights/np.sum(self.weights)

        # print (self.weights)
        particles_id = np.random.choice(a=self.num_particles, size=self.num_particles, p=self.weights[:, 0])

        # print (particles_id)

        particles = self.particles[particles_id]

        # print (particles.shape)

        return particles

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """

        # resample
        self.particles = self.resample_particles()

        self.frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # update particles
        u_noise = np.zeros(self.num_particles)
        v_noise = np.zeros(self.num_particles)
        cv2.randn(u_noise, 0, self.sigma_dyn)
        cv2.randn(v_noise, 0, self.sigma_dyn)


        self.particles[:, 0] = self.particles[:, 0].T + u_noise
        self.particles[:, 1] = self.particles[:, 1].T + v_noise

        # calculate weight
        m = self.template_gray.shape[0]
        n = self.template_gray.shape[1]
        weight_list = []
        for i in range(self.num_particles):
            col, row = self.particles[i]

            frame_cutout = self.frame_gray[(row-int(np.floor(m/2))):(row+int(np.ceil(m/2))), (col-int(np.floor(n/2))):(col+int(np.ceil(n/2)))]

            err = self.get_error_metric(self.template_gray, frame_cutout)
            weight = np.exp(-1 * err / (2 * np.square(self.sigma_exp)))
            weight_list.append(weight)

        self.weights[:, 0] = weight_list/np.sum(weight_list)


    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            # print(self.particles[i, 0], self.weights[i])
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            # print(x_weighted_mean)
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        # Complete the rest of the code as instructed.

        x_weighted_mean = int(x_weighted_mean)
        y_weighted_mean = int(y_weighted_mean)

        # print ((x_weighted_mean, y_weighted_mean))
        m = self.template_gray.shape[0]*self.best_scale/1000
        n = self.template_gray.shape[1]*self.best_scale/1000
        # print(m, n)
        # print((x_weighted_mean-np.floor(n/2), y_weighted_mean-np.floor(m/2)), (x_weighted_mean+np.ceil(n/2), y_weighted_mean+np.ceil(m/2)))
        cv2.rectangle(frame_in, (x_weighted_mean-int(np.floor(n/2)), y_weighted_mean-int(np.floor(m/2))), (x_weighted_mean+int(np.ceil(n/2)), y_weighted_mean+int(np.ceil(m/2))), (0, 255, 0), 4)

        for x, y, _ in self.particles:
            cv2.circle(frame_in, (x, y), 1, (255, 255, 0), 1)

        d_weighted_mean = 0
        for i in range(self.num_particles):
            d_weighted_mean += np.linalg.norm(self.particles[i, :2]-[x_weighted_mean, y_weighted_mean])*self.weights[i]
        # print (d_weighted_mean)

        cv2.circle(frame_in, (x_weighted_mean, y_weighted_mean), int(d_weighted_mean), (0, 0, 255), 3)


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """

        # resample
        self.particles = self.resample_particles()

        self.frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # update particles
        u_noise = np.zeros(self.num_particles)
        v_noise = np.zeros(self.num_particles)
        cv2.randn(u_noise, 0, self.sigma_dyn)
        cv2.randn(v_noise, 0, self.sigma_dyn)

        self.particles[:, 0] = self.particles[:, 0].T + u_noise
        self.particles[:, 1] = self.particles[:, 1].T + v_noise

        # calculate weight
        m = self.template_gray.shape[0]
        n = self.template_gray.shape[1]
        weight_list = []
        for i in range(self.num_particles):
            col, row = self.particles[i]

            frame_cutout = self.frame_gray[(row-int(np.floor(m/2))):(row+int(np.ceil(m/2))), (col-int(np.floor(n/2))):(col+int(np.ceil(n/2)))]

            err = self.get_error_metric(self.template_gray, frame_cutout)
            weight = np.exp(-1 * err / (2 * np.square(self.sigma_exp)))
            weight_list.append(weight)

        self.weights[:, 0] = weight_list/np.sum(weight_list)

        # update template
        best_idx = np.argmax(self.weights[:, 0])
        best_col, best_row = self.particles[best_idx]

        best = self.frame_gray[(best_row-int(np.floor(m/2))):(best_row+int(np.ceil(m/2))), (best_col-int(np.floor(n/2))):(best_col+int(np.ceil(n/2)))]
        # print ()
        # print (self.template_gray.shape)
        # print (best.shape)
        self.template_gray = (1 - self.alpha)*self.template_gray + self.alpha*best



class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.scale = kwargs.get('scale', 0.999)
        self.sigma_sca = kwargs.get('sigma_sca', 3)


        self.previous_best_err = 10000


        self.particles = np.c_[
            np.random.randint(int(self.template_rect['x']), int(self.template_rect['x'] + self.template_rect['w']),
                              self.num_particles),
            np.random.randint(int(self.template_rect['y']), int(self.template_rect['y'] + self.template_rect['h']),
                              self.num_particles),
            np.random.randint(self.scale*1000, 1001, self.num_particles)]  # Initialize your particles array. Read the docstring.

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """

        # resample
        self.particles = self.resample_particles()

        self.frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # update particles
        u_noise = np.zeros(self.num_particles)
        v_noise = np.zeros(self.num_particles)
        w_noise = np.zeros(self.num_particles)
        cv2.randn(u_noise, 0, self.sigma_dyn)
        cv2.randn(v_noise, 0, self.sigma_dyn)
        cv2.randn(w_noise, 0, self.sigma_sca)

        # print(w_noise)

        # self.particles[:, 2] = self.best_scale
        self.particles[:, 0] = self.particles[:, 0].T + u_noise
        self.particles[:, 1] = self.particles[:, 1].T + v_noise
        self.particles[:, 2] = self.particles[:, 2].T + w_noise
        #
        self.particles[:, 2][self.particles[:, 2] < 100] = 100

        # calculate weight

        weight_list = []
        err_list = []

        for i in range(self.num_particles):
            col, row, scale = self.particles[i]

            w = int(self.template_gray.shape[1] * scale / 1000)
            h = int(self.template_gray.shape[0] * scale / 1000)

            template_gray_scale_temp = cv2.resize(self.template_gray, (w, h), interpolation=cv2.INTER_CUBIC)

            m = template_gray_scale_temp.shape[0]
            n = template_gray_scale_temp.shape[1]

            frame_cutout = self.frame_gray[(row - int(np.floor(m / 2))):(row + int(np.ceil(m / 2))),
                           (col - int(np.floor(n / 2))):(col + int(np.ceil(n / 2)))]

            err = self.get_error_metric(template_gray_scale_temp, frame_cutout)
            weight = np.exp(-1 * err / (2 * np.square(self.sigma_exp)))
            err_list.append(err)
            weight_list.append(weight)

        # self.weights[:, 0] = weight_list/np.sum(weight_list)
        # print(weight_list)

        weight_list_ave = weight_list / np.sum(weight_list)
        # print(weight_list_ave)

        # update template
        best_idx = np.argmax(weight_list_ave)
        best_err = err_list[best_idx]

        # print (best_err)
        best_col, best_row, best_scale = self.particles[best_idx]
        # print(best_scale)
        # print("best err = ", best_err)

        if best_err - self.previous_best_err <= 800:
            # print ("yes")

            self.previous_best_err = best_err
            if np.sum(weight_list) != 0:
                self.weights[:, 0] = weight_list_ave
            else:
                self.weights = np.full((self.num_particles, 1), 1.0 / self.num_particles)

            self.best_scale = best_scale



