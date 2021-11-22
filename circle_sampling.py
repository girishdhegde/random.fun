import random
import math

import numpy as np 
import cv2 
import matplotlib.pyplot as plt 


__author__ = '__Girish_Hegde__'


class Sampler:
    def __init__(self, radius=1, center=(0, 0), method='rejection_sample'):
        self.radius = radius
        self.r2 = radius**2
        self.cx, self.cy = center
        self._sample_unit_circle = getattr(self, method)

    def sample(self, n=1):
        points = []
        for i in range(n):
            x, y = self._sample_unit_circle()
            # translate (x, y) 
            x, y = self.cx + x*self.radius, self.cy + y*self.radius
            points.append((x, y))
        return np.array(points)

    def rejection_sample(self):
        """Rejection Sampling
        
        1. point = uniform()
        2. if point satisfies required equation: return point
        3. else: repeat 1, 2, 3

        For circle:
            eqution: x2 + y2 = r2
            prob(success) = area(circle)/area(square)
                          = (pi.r2)/(4r2)
                          = pi/4 = 78.5%
            average trials per sample = 4/pi = 1.28

        """
        x = (random.random() - 0.5)*2
        y = (random.random() - 0.5)*2
        # check if (x, y) inside unit circle
        while not x*x + y*y < 1:
            # uniform(-1, 1)
            x = (random.random() - 0.5)*2
            y = (random.random() - 0.5)*2
        return (x, y)

    def ITS(self):
        """Inverse Transform Samples

        1. Get pdf
        2. cdf = integration(pdf)
        3. Get inverse cdf
        3. sample = inverse_cdf(uniform sample)

        For circle:
            polar coords -> (r, theta)
            theta = uniform(0, 2.pi)
            radius sampling:
                r != uniform(0, 1) <- as radius increases perimeter increases 
                hence uniform sampling -> samples(large radius) < samples(small radius)
                -----------------------------------------------------------------------
                samples = k.perimeter = k.2.pi.r
                pdf = mr
                integral(pdf) = prob(samples space) = 1
                mr2/2 = 1
                m = 2
                pdf = 2r
                cdf = integral(pdf) = r2
                r = sqrt(uniform(0, 1)) <- inverse transform

        """
        # uniform_sample(0, 2.pi)
        theta = random.random()*2*math.pi
        # r = sqrt(uniform(0, 1)) <- inverse transform
        radius = math.sqrt(random.random())
        # polar -> cartesian
        x = radius*math.cos(theta)
        y = radius*math.sin(theta)
        return (x, y)


    def triangle_sample(self):
        """ Zero Area Triangle Sampling

        1. consider a parallellogram
        2. v1 = sample a vector along 1 side
        3. v2 = sample a vector along adjacent side 
        4. point inside ||gm = v1 + v2
        5. point inside triangle = v1 + v2 if inside diagonal else reflect v1 + v2

        For circle:
            consider circle is made up of infinite triangles of area zero -> lines
            sample = uniform(0, 1) + uniform(0, 1)
            radius = sample  if sample < 1 else 2 - sample
            choose triangle -> theta = uniform(0, 2.pi)
            Note: pdf(of this method) == pdf(ITS method above) = 2r 

        """ 
        # uniform_sample(0, 2.pi)
        theta = random.random()*2*math.pi
        radius = random.random() + random.random()
        if radius >= 1:
            radius = 2 - radius
        # polar -> cartesian
        x = radius*math.cos(theta)
        y = radius*math.sin(theta)
        return (x, y)


def main():
    r = 200
    # cx, cy = -200, 100
    cx, cy = 0, 0
    n = 10000
    clr = (0, 0, 255)

    hw = 3*(max(abs(cx), abs(cy)) + r)
    if hw%2 == 0:
        hw += 1
    coord_frame = np.zeros((hw, hw, 3),  dtype=np.uint8)
    coord_frame[hw//2, :] = 255
    coord_frame[:, hw//2] = 255
    cv2.circle(coord_frame, (hw//2  + cx, hw - (hw//2 +cy)), r, (255, 0, 0))

    sampler = Sampler(r, (cx, cy),  'ITS')
    # sampler = Sampler(r, (cx, cy),  'rejection_sample')
    # sampler = Sampler(r, (cx, cy),  'triangle_sample')
    
    samples = sampler.sample(n).astype(np.int32)
    for pt in  samples: 
        coord_frame[hw - (hw//2 + pt[1]), hw//2 + pt[0]] = clr
        cv2.imshow("circle sampling", coord_frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    # coord_frame[hw - 1 - (hw//2 + samples[:, 1]), hw//2 + samples[:, 0]] = [0, 0, 255]
    # cv2.imshow("circle sampling", coord_frame)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()