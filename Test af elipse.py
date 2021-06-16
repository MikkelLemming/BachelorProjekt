import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from scipy import optimize as opt
import math as m
import statistics as stat

# ---------------------------------------------- Deffining the data -------------------------------------------------
data_S62 = np.genfromtxt("PosData_S62.txt", unpack=True, skip_header=1)
data_S62 = [data_S62[0], data_S62[1]*10**-3, data_S62[3]*10**-3, data_S62[2]*10**-3, data_S62[4]*10**-3]
x_S62 = data_S62[1]
y_S62 = data_S62[2]

data_S2 = np.genfromtxt("PosData_S2.txt", unpack=True, skip_header=1)
x_S2 = data_S2[1]
y_S2 = data_S2[2]

data_S38 = np.genfromtxt("PosData_S38.txt", unpack=True, skip_header=1)
x_S38 = data_S38[1]
y_S38 = data_S38[2]

data_S55 = np.genfromtxt("PosData_S55.txt", unpack=True, skip_header=1)
x_S55 = data_S55[1]
y_S55 = data_S55[2]


# ---------------------------------------------- Ellipse fitting 1 (Fail so far)------------------------------------
def elipse_fit(x_coord, y_coord):
    xmean, ymean = x_coord.mean(), y_coord.mean()
    
    x_coord -= xmean
    y_coord -= ymean
    
    
    U, S, V = np.linalg.svd(np.stack((x_coord,y_coord)))
   
    tt = np.linspace(0,2*np.pi, 1000)
    circle = np.stack((np.cos(tt), np.sin(tt)))
    transform = np.sqrt(2/len(x_coord))*U.dot(np.diag(S))
    fit = transform.dot(circle) + np.array([[xmean],[ymean]])
    plt.plot(fit[0,:], fit[1,:], 'r')
    plt.show()

#elipse_fit(x_S2, y_S2)
#elipse_fit(x_S38, y_S38)
#elipse_fit(x_S55, y_S55)

plt.plot(x_S62, y_S62, 'o', color='red')
plt.plot(x_S2, y_S2, 'o', color='blue')
plt.plot(x_S38, y_S38, 'o', color='green')
plt.plot(x_S55, y_S55, 'o', color='yellow')
plt.plot(0,0, '*', color='black')


plt.xlim([-0.4, 0.2])
plt.ylim([-0.2,0.2])
plt.xlabel("R.A. ('""')")
plt.ylabel("Dec. ('""')")
plt.legend(["S62", "S2", "S38", "S0-102/S55", "Sgr A*"])


# -------------------------------------- Converting to r and theta instead of x and y-------------------------------
def r_theta_data(data, x_F = 0, y_F = 0):
    r = []
    theta = []
    
    for i in range(len(data[1])):
        r.append(np.sqrt((data[1][i] - x_F)**2+(data[2][i] - y_F)**2))
        if data[1][i] - x_F == 0:
            theta.append(1/2*np.pi)
        elif data[1][i] - x_F > 0:
            theta.append(np.arctan((data[2][i] - y_F)/(data[1][i] - x_F)))
        elif data[1][i] - x_F < 0:
            theta.append(np.pi + np.arctan((data[2][i] - y_F)/(data[1][i] - x_F)))
        else: 
            theta.append('NA')
            
    return r, theta




#def diff(r, theta, f_t): #f_t = fine tuning
#    diff_list = []
#    angle_list = []
#    angle_correction_list = []
#        
#    for i in range(360*f_t):
#        angle = theta + np.radians(i/f_t)
#            
#        for i2 in range(len(angle)):
#            if angle[i2] > 2*np.pi:
#                angle[i2] = angle[i2] - 2*np.pi
#            if angle[i2] < 0:
#                angle[i2] = angle[i2] + 2*np.pi
#        count = sum(map(lambda x: round(np.degrees(x)) in range(90, 270), angle))
#            
#        if count < 1/2 * len(angle):
#            diff_array = ([i**5 for i in r]*np.sin(angle))**2
    #        diff_array = (r*np.sin(angle))**2
#            diff = sum(diff_array)
#            diff_list.append(diff)
#            angle_list.append(angle)
#            angle_correction_list.append(np.radians(i/f_t))
#           
#    return angle_correction_list[diff_list.index(min(diff_list))]  

# An alternitive way to optimize the angle 
def diff(r, theta, f_t): 
    angle_diff = theta[r.index(max(r))]
    return -angle_diff


def r_theta_data_corrected(data, f_t = 2): #Turning the ellipse for easyer fit
    r = r_theta_data(data)[0]
    theta = r_theta_data(data)[1]
    d = diff(r, theta, f_t)
    new_theta = theta + d
    
    return r, new_theta%(2*np.pi)
    
    

def f_ellipse(theta, p): #p = (a,e)
    a,e,theta_0 = p
    return a*(1-e**2)/(1-e*np.cos(theta - theta_0))

def residual(p, r, theta):
    return r - f_ellipse(theta, p)

def jac(p, r, theta):
    a,e,theta_0 = p
    da = (1-e**2)/(1-e*np.cos(theta-theta_0))
    de = (a*(1-e**2)*np.cos(theta-theta_0) - 2*e*a*(1-e*np.cos(theta-theta_0)))/(1-e*np.cos(theta-theta_0))**2
    #dt = a*((1+e**2)*np.cos(theta - theta_0) - 2*e)/(1-e*np.cos(theta - theta_0))^2
    dt = 0*theta
    
    return -da, -de, -dt
    return np.array((-da,-de,-dt)).T
    
#Initial guess:
p0 = (0.1,0.5, 1/2*np.pi)

def plsq(data, p_init=p0):
    r = r_theta_data_corrected(data)[0]
    theta = r_theta_data_corrected(data)[1]
    
    plsq = opt.leastsq(residual, p_init, Dfun=jac, args=(r,theta), col_deriv=True)
    return plsq

def plot_ellipse(data, p_init=p0):
    plt.figure()
    print(plsq(data, p_init))
    pl.polar(r_theta_data_corrected(data)[1], r_theta_data_corrected(data)[0], 'x')
    theta_grid = np.linspace(0, 2*np.pi, 200)
    pl.polar(theta_grid, f_ellipse(theta_grid, plsq(data, p_init)[0]), lw=2)
    pl.show()

plot_ellipse(data_S2)
plot_ellipse(data_S38)
plot_ellipse(data_S55)


p = (plsq(data_S2, p0)[0][0], plsq(data_S2, p0)[0][1])



#-------------------------------- Finding the precission of the fit -------------------------------------------------
def fit_precision(data, f_t = 11, p_init = p0, test_interval = 1/4*np.pi):
    sq_dist_list = []
    d = r_theta_data_corrected(data)
    p = (plsq(data, p_init)[0][0], plsq(data, p_init)[0][1])
    
    for i in range(len(d[0])):
        sq_dist = []
        for x in np.linspace(d[1][i]-test_interval, d[1][i]+test_interval, f_t):
            sq_dist.append( (d[0][i]*np.sin(d[1][i]) - f_ellipse(x,p)*np.sin(x))**2 + (d[0][i]*np.cos(d[1][i]) - f_ellipse(x,p)*np.cos(x))**2 )
        sq_dist_list.append(min(sq_dist))
        
    return sum(np.sqrt(sq_dist_list))/len(data[0])


print("\nThe precission of S2 fit is:")
print(fit_precision(data_S2))

print("\nThe precission of S38 fit is:")
print(fit_precision(data_S38))

print("\nThe precission of S55 fit is:")
print(fit_precision(data_S55))




#---------------------- Finding the ratio between swept area and time passed ----------------------------------------
def r_theta_center(data, p_init = p0): 
    r, theta = r_theta_data_corrected(data)
    p = plsq(data, p_init)[0]
    x = []
    y = []

    for i in range(len(r)):
        x.append(r[i]*np.cos(theta[i]))
        y.append(r[i]*np.sin(theta[i]))
    
    new_data = [[], x, y]
    dist_to_center = f_ellipse(0, p) - p[0] # Can also be calculated as e*a
    
    r_corrected, theta_corrected = r_theta_data(new_data, x_F = dist_to_center)


    return [p,r_corrected ,[i%(2*np.pi) for i in theta_corrected], dist_to_center]


def ellipse_area(r, theta, p):
    if m.floor(theta/(1/2*np.pi)) == 0:
        area = 1/2*p[0]**2*np.sqrt(1-p[1]**2)*( m.floor(theta/(1/2*np.pi))*np.pi + np.arctan(np.tan(theta)/(np.sqrt(1-p[1]**2))) )
    
    elif m.floor(theta/(1/2*np.pi)) == 1 or m.floor(theta/(1/2*np.pi)) == 2:
        area = 1/2*p[0]**2*np.sqrt(1-p[1]**2)*(np.pi + np.arctan(np.tan(theta)/(np.sqrt(1-p[1]**2))) )
        
    elif m.floor(theta/(1/2*np.pi)) == 3:
        area = 1/2*p[0]**2*np.sqrt(1-p[1]**2)*(2*np.pi + np.arctan(np.tan(theta)/(np.sqrt(1-p[1]**2))) )
    
    area = area - 1/2*p[0]*r*p[1]*np.sin(theta)
    
    return area
    
    

def ellipse_cutout_area(r_start, r_end, theta_start, theta_end, p):
    a, e = p
    
#    area_start = 1/2 * a**2 * np.sqrt(1-e**2) * ( theta_start - e*np.sin(theta_start) )
#    area_end = 1/2 * a**2 * np.sqrt(1-e**2) * ( theta_end - e*np.sin(theta_end) )
    
    
#    area_start = 1/2 * a * ( a*np.sqrt(1-e**2)*theta_start - r_start*e*np.sin(theta_start) )
#    area_end = 1/2 * a * ( a*np.sqrt(1-e**2)*theta_end - r_end*e*np.sin(theta_end) )

#    area_start = 1/2 * a * ( a*np.sqrt(1-e**2)*np.arctan(np.tan(theta_start)/np.sqrt(1-e**2)) - r_end*e*np.sin(theta_start) )
#    area_end = 1/2 * a * ( a*np.sqrt(1-e**2)*np.arctan(np.tan(theta_end)/np.sqrt(1-e**2)) - r_end*e*np.sin(theta_end) )

    area_start = ellipse_area(r_start, theta_start, p)
    area_end = ellipse_area(r_end, theta_end, p)

    area_tot = np.pi*a**2 * np.sqrt(1-e**2)

    if theta_start < theta_end:
        area = area_end - area_start

    if theta_start > theta_end:
        area = area_tot - (area_end - area_start)
        
    if theta_start == theta_end:
        print("Two data points seems to be overlaping, one the latter will not be considerd.")
        area = 0
        
    return area

def ellipse_area_ratios(data):
    p, r, theta, dist_to_center = r_theta_center(data)
    areas = []
    times = []
    ratios = []
    
    for i in range(len(theta)-1):
        area = ellipse_cutout_area(r[i], r[i+1], theta[i], theta[i+1], p)
        areas.append(area)
        
    for i in range(len(data[0])-1):
        time = data[0][i+1] - data[0][i]
        times.append(time)

    for i in range(len(areas)):
        ratio = areas[i]/times[i]
        ratios.append(ratio)
        
    time_test = data[0][-1] - data[0][0]
    area_test = ellipse_cutout_area(r[0], r[-1], theta[0], theta[-1], p)
    ratio_test = area_test/time_test
    
        
    return ratios, areas, times, ratio_test




def ellipse_area_time_ratios(data):
    x = data[1]
    y = data[2]
    t = data[0]
    
    areas = []
    times = []
    ratios = []
    
    for i in range(len(x)-1):
        area = 1/2 * abs(x[i]*y[i+1] - x[i+1]*y[i])
        areas.append(area)
        
    for i in range(len(t)-1):
        time = t[i+1] - t[i]
        times.append(time)
        
    for i in range(len(areas)):
        ratio = areas[i]/times[i]
        ratios.append(ratio)
        
    area_test = 1/2*abs(x[0]*y[-1] - x[-1]*y[0])
    time_test = t[-1] - t[0]
    ratio_test = area_test/time_test
        
    return ratios, areas, times, ratio_test





































# ----------------------------------------------Testing of stuff ---------------------------------------------------
p, r, theta, dist = r_theta_center(data_S2)
x = []
y = []
for i in range(len(r)):
    x.append(r[i]*np.cos(theta[i]))
    y.append(r[i]*np.sin(theta[i]))


r, theta = r_theta_data_corrected(data_S2)
x2 = []
y2 = []
for i in range(len(r)):
    x2.append(r[i]*np.cos(theta[i]))
    y2.append(r[i]*np.sin(theta[i]))
    


plt.figure()
plt.title("The S2 orbit; original, corrected and centerd")
plt.plot(x,y,'*')
plt.plot(x2,y2,'x')
plt.plot(x_S2, y_S2, 'o')
plt.plot(0,0,'^', color='red')
plt.plot(dist, 0, 'X')




data_S62_new = np.genfromtxt("PosData_S62_new.txt", unpack=True, skip_header=1)
data_S62_new = [data_S62_new[0], data_S62_new[1]*10**-3, data_S62_new[3]*10**-3, data_S62_new[2]*10**-3, data_S62_new[4]*10**-3]






#---------- Bestemmelse af perioden -------------------------------------------
e = ellipse_area_time_ratios(data_S2)[0]
e_2 = ellipse_area_ratios(data_S2)

E = []

for i in range(len(e_2[1])):
    if e_2[1][i] < np.pi*p[0]**2*np.sqrt(1-p[1]**2):
        E.append(e_2[0][i])

avg_e = sum(e)/len(e)
avg_E = sum(E)/len(E)

K = np.pi*p[0]**2*np.sqrt(1-p[1]**2)

here_1 = K / (avg_e)
here_2 = K / (avg_E)
here_3 = K/ellipse_area_time_ratios(data_S2)[3]
here_4 = K/e_2[3]


def print_stuff():
    print(here_1)
    print(here_2)
    print(here_3)
    print(here_4)









def f_ellipse_2 (theta, p):
    a,e = p
#    x^2 / a + y^2 / (a*np.sqrt(1-e**2)) = 1
    
    
    
    
    
import numpy as np
import matplotlib.pyplot as plt
alpha = 5
beta = 3
N = 500
DIM = 2

#np.random.seed(2)

# Generate random points on the unit circle by sampling uniform angles
#theta = np.random.uniform(0, 2*np.pi, (N,1))
#eps_noise = 0.2 * np.random.normal(size=[N,1])
#circle = np.hstack([np.cos(theta), np.sin(theta)])

# Stretch and rotate circle to an ellipse with random linear tranformation
#B = np.random.randint(-3, 3, (DIM, DIM))
#noisy_ellipse = circle.dot(B) + eps_noise

X = []
for i in range(len(x_S2)):
    X.append([x_S2[i]])
X = np.array(X)


Y = []
for i in range(len(y_S2)):
    Y.append([y_S2[i]])
Y = np.array(Y)


plt.figure()

# Extract x coords and y coords of the ellipse as column vectors
#X = noisy_ellipse[:,0:1]
#Y = noisy_ellipse[:,1:]

# Formulate and solve the least squares problem ||Ax - b ||^2
A = np.hstack([X**2, X * Y, Y**2, X, Y])
b = np.ones_like(X)
x = np.linalg.lstsq(A, b)[0].squeeze()

# Print the equation of the ellipse in standard form
print('The ellipse is given by {0:.3}x^2 + {1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1'.format(x[0], x[1],x[2],x[3],x[4]))

# Plot the noisy data
plt.scatter(X, Y, label='Data Points')

# Plot the original ellipse from which the data was generated
phi = np.linspace(0, 2*np.pi, 1000).reshape((1000,1))
c = np.hstack([np.cos(phi), np.sin(phi)])
ground_truth_ellipse = c.dot(B)
plt.plot(ground_truth_ellipse[:,0], ground_truth_ellipse[:,1], 'k--', label='Generating Ellipse')

# Plot the least squares ellipse
x_coord = np.linspace(-5,5,300)
y_coord = np.linspace(-5,5,300)
X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord
plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)

plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

