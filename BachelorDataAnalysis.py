import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from scipy import optimize as opt
import math as m
from numpy.linalg import eig, inv, svd
from mpl_toolkits.mplot3d import Axes3D

#---------------------------------------------------------------------------------------------------------
# ---------------------------------------------- Deffining the data -------------------------------------------------
#---------------------------------------------------------------------------------------------------------
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



data_S62_new = np.genfromtxt("PosData_S62_new.txt", unpack=True, skip_header=1)
data_S62_new = [data_S62_new[0], data_S62_new[1]*10**-3, data_S62_new[3]*10**-3, data_S62_new[2]*10**-3, data_S62_new[4]*10**-3]
x_S62_new = data_S62_new[1]
y_S62_new = data_S62_new[2]

#data_S2_New = np.genfromtxt("PosData_S2_New.txt", unpack=True, skip_header=1)
#x_S2_New = data_S2_New[1]
#y_S2_New = data_S2_New[2]

#---------------------------------------------------------------------------------------------------------
# ---------------------------------------------- Ellipse fitting 1 (Fail so far) ------------------------------------
#---------------------------------------------------------------------------------------------------------
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

plt.plot(x_S62, y_S62, '*', color='red', markersize=5)
plt.plot(x_S2, y_S2, '*', color='blue', markersize=5)
plt.plot(x_S38, y_S38, '*', color='green', markersize=5)
plt.plot(x_S55, y_S55, '*', color='orange', markersize=5)
plt.plot(0,0, 'o', color='black', markersize=7)


plt.xlim([-0.3, 0.2])
plt.ylim([-0.2,0.2])
plt.xlabel("$\Delta$ R.A. ('""')")
plt.ylabel("$\Delta$ Dec. ('""')")
plt.legend(["S62", "S2", "S38", "S55", "Sgr A*"])


#---------------------------------------------------------------------------------------------------------
# -------------------------------------- Converting to r and theta instead of x and y-------------------------------
#---------------------------------------------------------------------------------------------------------
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
def diff(r, theta, f_t=2): 
    angle_diff = theta[r.index(max(r))]
    return -angle_diff


def r_theta_data_corrected(data, f_t = 2): #Turning the ellipse for easyer fit
    r = r_theta_data(data)[0]
    theta = r_theta_data(data)[1]
    d = diff(r, theta, f_t)
    new_theta = theta + d
    
    return r, new_theta%(2*np.pi)
    
    

def f_ellipse(theta, p, phi=0): #p = (a,e)
    a,e = p
    return a*(1-e**2)/(1-e*np.cos(theta - phi))

def residual(p, r, theta, phi=0):
    return r - f_ellipse(theta, p, phi)

def jac(p, r, theta):
    a,e = p
    da = (1-e**2)/(1-e*np.cos(theta))
    de = (a*(1-e**2)*np.cos(theta) - 2*e*a*(1-e*np.cos(theta)))/(1-e*np.cos(theta))**2
    
    return -da, -de
    return np.array((-da,-de)).T
    
#Initial guess:
p0 = (0.1,0.5)

def plsq(data, p_init=p0):
    r = r_theta_data_corrected(data)[0]
    theta = r_theta_data_corrected(data)[1]
    
    plsq = opt.leastsq(residual, p_init, Dfun=jac, args=(r,theta), col_deriv=True)
    
#    plsq[0][0] = plsq[0][0]/np.sin(np.radians(133-90))
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
plot_ellipse(data_S62)

p = (plsq(data_S2, p0)[0][0], plsq(data_S2, p0)[0][1])


#---------------------------------------------------------------------------------------------------------
#-------------------------------- Finding the precission of the fit -------------------------------------------------
#---------------------------------------------------------------------------------------------------------
def fit_precision1(data, f_t = 100, p_init = p0, test_interval = 1/8*np.pi):
    sq_dist_list = []
    d = r_theta_data_corrected(data)
    p = (plsq(data, p0)[0][0], plsq(data, p0)[0][1])
    
    for i in range(len(d[0])):
        sq_dist = []
        for x in np.linspace(d[1][i]-test_interval, d[1][i]+test_interval, f_t):
            sq_dist.append( (d[0][i]*np.sin(d[1][i]) - f_ellipse(x,p)*np.sin(x))**2 + (d[0][i]*np.cos(d[1][i]) - f_ellipse(x,p)*np.cos(x))**2 )
        sq_dist_list.append(min(sq_dist))
        
    return sum(np.sqrt(sq_dist_list))/len(data[0])


print("\nThe precission of S2 fit is:")
print(fit_precision1(data_S2))

print("\nThe precission of S38 fit is:")
print(fit_precision1(data_S38))

print("\nThe precission of S55 fit is:")
print(fit_precision1(data_S55))

print("\nThe precission of S62 fit is:")
print(fit_precision1(data_S62))


#---------------------------------------------------------------------------------------------------------
#---------------------- Finding the ratio between swept area and time passed ----------------------------------------
#---------------------------------------------------------------------------------------------------------
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
    theta_f = theta
    theta_c = m.atan2( (r*np.sin(theta_f)) , (r*np.cos(theta_f) - p[1]*p[0]) )
    
    theta_f = theta_f%(2*np.pi)
    theta_c = theta_c%(2*np.pi)


#    if theta_c == 2*np.pi:
#        theta_c = theta - 0.000000000001
    
#    theta_r = theta_c%(1/2*np.pi)
    
#    if m.floor(theta/(1/2*np.pi)) == 0:
#        area = 1/2*p[0]**2*np.sqrt(1-p[1]**2)*( np.arctan(np.tan(theta)/(np.sqrt(1-p[1]**2))) )
    
#    elif m.floor(theta/(1/2*np.pi)) == 1 or m.floor(theta/(1/2*np.pi)) == 2:
#        area = 1/2*p[0]**2*np.sqrt(1-p[1]**2)*(np.pi + np.arctan(np.tan(theta)/(np.sqrt(1-p[1]**2))) )
        
#    elif m.floor(theta/(1/2*np.pi)) == 3:
#        area = 1/2*p[0]**2*np.sqrt(1-p[1]**2)*(2*np.pi + np.arctan(np.tan(theta)/(np.sqrt(1-p[1]**2))) )
    
#    area = area - 1/2*p[0]*r*p[1]*np.sin(theta)
    

#    area_from_center = 1/2*p[0]**2*np.sqrt(1-p[1]**2)*(np.arctan(np.tan(theta_r)/(np.sqrt(1-p[1]**2))) +  (theta_c-theta_r))
    area_from_center = 1/2*p[0]**2*np.sqrt(1-p[1]**2)* ( theta_c - np.arctan( (np.sqrt(1-p[1]**2)-1)*p[0]*np.sin(2*theta_c)/( p[0]*(1+np.sqrt(1-p[1]**2)) + (np.sqrt(1-p[1]**2)-1)*p[0]*np.cos(2*theta_c) ) ) )
   
    area_of_triangle = 1/2 * p[0]*p[1]* r * np.sin(theta_f)

    area = area_from_center + area_of_triangle
    
    return area, area_from_center, area_of_triangle, theta_c, theta_f
    
    

def ellipse_cutout_area(r_start, r_end, theta_start, theta_end, p):
#    a, e = p
    
#    area_start = 1/2 * a**2 * np.sqrt(1-e**2) * ( theta_start - e*np.sin(theta_start) )
#    area_end = 1/2 * a**2 * np.sqrt(1-e**2) * ( theta_end - e*np.sin(theta_end) )
    
    
#    area_start = 1/2 * a * ( a*np.sqrt(1-e**2)*theta_start - r_start*e*np.sin(theta_start) )
#    area_end = 1/2 * a * ( a*np.sqrt(1-e**2)*theta_end - r_end*e*np.sin(theta_end) )


#    area_start = 1/2 * a * ( a*np.sqrt(1-e**2)*np.arctan(np.tan(theta_start)/np.sqrt(1-e**2)) - r_end*e*np.sin(theta_start) )
#    area_end = 1/2 * a * ( a*np.sqrt(1-e**2)*np.arctan(np.tan(theta_end)/np.sqrt(1-e**2)) - r_end*e*np.sin(theta_end) )

    area_start = ellipse_area(r_start, theta_start, p)[0]
    area_end = ellipse_area(r_end, theta_end, p)[0]

    area_tot = np.pi*p[0]**2 * np.sqrt(1-p[1]**2)

    area = area_end - area_start

    if theta_start > np.pi and theta_end < np.pi:
        area = area_tot + (area_end - area_start)
        
    if theta_start == theta_end:
        print("Two data points seems to be overlaping, one the latter will not be considerd.")
        area = 0
        
    return area

def ellipse_area_ratios(data, a=0, e=0):
    r, theta = r_theta_data_corrected(data)
    p = plsq(data)[0]
    if a != 0:
        p = (a,e)
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








def ellipse_area_ratios_ristricted(data, theta_min=np.radians(10), a=0, e=0, theta=0, r=0):
    if theta == 0:
        r, theta = r_theta_data_corrected(data)
    p = plsq(data)[0]
    if a != 0:
        p = (a,e)
    
    i_list = []
    j_list = []
    
    for i in range(len(theta)):
        for j in range(i+1,len(theta)):
            if theta[j] - theta[i] >= theta_min:
              i_list.append(i)
              j_list.append(j)
              break
        
    
    areas = []
    times = []
    ratios = []
    
    for i in range(len(i_list)):
        area = ellipse_cutout_area(r[i_list[i]], r[j_list[i]], theta[i_list[i]], theta[j_list[i]], p)
#        area = ellipse_cutout_area(f_ellipse(theta[i_list[i]],p), f_ellipse(theta[j_list[i]],p), theta[i_list[i]], theta[j_list[i]], p)
        areas.append(area)
        
    for i in range(len(i_list)):
        time = data[0][j_list[i]] - data[0][i_list[i]]
        times.append(time)

    for i in range(len(areas)):
        ratio = areas[i]/times[i]
        ratios.append(ratio)
        
    time_test = data[0][-1] - data[0][0]
    area_test = ellipse_cutout_area(r[0], r[-1], theta[0], theta[-1], p)
    ratio_test = area_test/time_test
    
    avg_ratio = sum(ratios)/len(ratios)
    
    return ratios, areas, times, ratio_test, avg_ratio, j_list, i_list



#---------------------------------------------------------------------------------------------------------
# ------------------ Determin period and at last the mass -------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
def period(data, a=0, e=0):
    p = plsq(data)[0]
    if a != 0:
        p = (a,e)
    A = np.pi * p[0]**2 * np.sqrt(1-p[1]**2)
    ratio = ellipse_area_ratios_ristricted(data)[4] #The averige ratio
    
    P = A/ratio
    return P


                                    # DER ER PROBLEMER MED DENNEHER FOR ANDRE METODER END PLSQ
def BH_mass(data, a=0, e=0):
    P = period(data, a, e) * 365.25*24*60**2      #Years to seconds
    a = np.tan(np.radians(plsq(data)[0][0]/3600))*8.18*10**3*3.09*10**16   #Arcsec to meters
    G = 6.67*10**(-11)
    m_sun = 1.99*10**30
    
    m = 4*np.pi**2*a**3/(G*P**2)
    m_solar = m / m_sun

    return m_solar


print("\nThe period of S2 is:", period(data_S2))
print("The period of S38 is:", period(data_S38))
print("The period of S55 is:", period(data_S55))
print("The period of S62 is:", period(data_S62))



print("\nFrom S2 the mass of Sgr A* is estimated to be:", BH_mass(data_S2)*10**(-6),"10^6 M_solar")
print("From S38 the mass of Sgr A* is estimated to be:", BH_mass(data_S38)*10**(-6),"10^6 M_solar")
print("From S55 the mass of Sgr A* is estimated to be:", BH_mass(data_S55)*10**(-6),"10^6 M_solar")
print("From S62 the mass of Sgr A* is estimated to be:", BH_mass(data_S62)*10**(-6),"10^6 M_solar")



#---------------------------------------------------------------------------------------------------------
# ----------------------------- Visulising what i have done to the orbits  ---------------------------------------------------
#---------------------------------------------------------------------------------------------------------
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








#---------------------------------------------------------------------------------------------------------
#---------- Bestemmelse af perioden ----------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
data_S62_new = np.genfromtxt("PosData_S62_new.txt", unpack=True, skip_header=1)
data_S62_new = [data_S62_new[0], data_S62_new[1]*10**-3, data_S62_new[3]*10**-3, data_S62_new[2]*10**-3, data_S62_new[4]*10**-3]

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





def plot_fit(data, a=0, e=0):
    S = r_theta_data(data)
    d = diff(S[0], S[1])
    p = plsq(data)[0]
    p2 = np.array([a,e])
#    print(d)

    X = []
    Y = []
    
    X2 = []
    Y2 = []

    for i in np.linspace(0,2*np.pi, 720):
        theta = i + d
        r = f_ellipse(theta, p)
        
        x = r*np.cos(i)
        y = r*np.sin(i)
        
        X.append(x)
        Y.append(y)

        if a != 0:
            r2 = f_ellipse(theta, p2)
            
            X2.append(r2*np.cos(i))
            Y2.append(r2*np.sin(i))


    plt.figure()
    plt.plot(X, Y, color='green')
    plt.plot(data[1], data[2], '*', color='blue') 
    plt.plot(0,0,'o', color='black')
    
    if a != 0:
        plt.plot(X2, Y2, color='red')

#    plt.xlim([-0.4, 0.2])
#    plt.ylim([-0.2,0.2])
#    return X, Y

def arcsec_to_pc(a):
    R = np.tan(np.radians(a/3600))*8*10**3*3.09*10**16 
    return R, R/(3.09*10**16)













#---------------------------------------------------------------------------------------------------------
# ---------------- Metode 2 ----------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
def __fit_ellipse(x, y):
    x, y = x[:, np.newaxis], y[:, np.newaxis]
    D = np.hstack(( x*x, x*y, y*y, x, y, np.ones_like(x) ))
    S, C = np.dot(D.T, D), np.zeros([6,6])
    C[0,2], C[2,0], C[1,1] = 2,2,-1
    U, s, V = svd(np.dot(inv(S), C))
    a = U[:,0]
    
    return a

def ellipse_center(a):
    b, c, d, f, g, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    
    num = b*b-a*c
    x0 = (c*d-b*f)/num
    y0 = (a*f-b*d)/num
    
    return np.array([x0, y0])

def ellipse_axis_length(a):
    b, c, d, f, g, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    
    up = 2*(a*f*f + c*d*d + g*b*b - 2*b*d*f - a*c*g)
    down1 = (b*b-a*c) * ((c-a)*np.sqrt(1+4*b*b/( (a-c)*(a-c) )) -(a+c) )
    down2 = (b*b-a*c) * ((a-c)*np.sqrt(1+4*b*b/( (a-c)*(a-c) )) -(a+c) )
    
    res1 = np.sqrt(up/down1)
    res2 = np.sqrt(up/down2)    
    
    return np.array([res1, res2])

def ellipse_angle_of_rotation(a):
    b, c, d, f, g, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return m.atan2(2*b,(a-c))/2
    
def fit_ellipse(x,y):
    a = __fit_ellipse(x,y)

    center = ellipse_center(a)
    phi = ellipse_angle_of_rotation(a)
    M,m = ellipse_axis_length(a)

    if m > M:
        M,m = m,M
        
    phi -= 2*np.pi*int(phi/(2*np.pi))
    return [M, m, center[0], center[1], phi]    




def ellipse_fit_2(data):
    x = data[1]
    y = data[2]
    
    fit = fit_ellipse(x,y)
    
    a = fit[0]
    e = np.sqrt(1 - (fit[1]/fit[0])**2)
    phi = fit[-1]
    c_x = fit[2]
    c_y = fit[3]
    
    return a, e, phi, c_x, c_y



def ellipse_plot_method2(data):
    arc = 2
    R = np.arange(0,arc*np.pi, 0.01)

    x = data[1]
    y = data[2]
    
    M, m, c_x, c_y, phi = fit_ellipse(x,y)
    
    xx = c_x + M*np.cos(R)*np.cos(phi) - m*np.sin(R)*np.sin(phi)
    yy = c_y + M*np.cos(R)*np.sin(phi) + m*np.sin(R)*np.cos(phi)
    
    
#    plt.plot(x,y, '*', color='blue')
    plt.plot(xx,yy, color='red')
    
    return xx, yy
    






#---------------------------------------------------------------------------------------------------------
# ------------------- My own minimized function ---------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
def sqr_min_fit(data, XY = "False"):
    
    def sqr_sum_f_ellipse(parameters):
        r, theta = r_theta_data(data)    
        a, e, phi = parameters
        
        residuals = []
        
        for i in range(len(r)):
            R = residual((a,e), r[i], theta[i], phi)
            residuals.append(R**2)
            
        return sum(residuals)
    
    guess_init = [0.1, 0.5, np.pi]
    
    bnds = ((0,1), (0.01,0.99), (0,2*np.pi))
    
    result = opt.minimize(sqr_sum_f_ellipse, guess_init, method='L-BFGS-B', bounds=bnds)    
    
    X1 = []
    Y1 = []
        
    p = (result.x[0], result.x[1])
    d = -result.x[2]
        
    for i in np.linspace(0, 2*np.pi, 720):
        theta = i + d
        r = f_ellipse(theta, p)
        x = r*np.cos(i)
        y = r*np.sin(i)
            
        X1.append(x)
        Y1.append(y)
    
    
    plt.plot(X1, Y1, color='orange')
    
    if XY == "True":
        return X1, Y1
    else:
        return result.x
    
#              !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! NEEDS WORK
#---------------------------------------------------------------------------------------------------------
# ------------------- Comparing the methods ---------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------    
def compair_methods(data, err = "no", method = [1,2,3]):
    X1 = []
    Y1 = []
    
    p = plsq(data)[0]
    S = r_theta_data(data)
    d = diff(S[0], S[1])
    
    for i in np.linspace(0, 2*np.pi, 720):
        theta = i + d
        r = f_ellipse(theta, p)
        x = r*np.cos(i)
        y = r*np.sin(i)
        
        X1.append(x)
        Y1.append(y)
        
    xerr = np.array(data[3])
    yerr = np.array(data[4])


    plt.figure()
    plt.plot(0,0,'o', color='black')
    
    if err == "no":
        plt.plot(data[1], data[2], '*', color='blue')
  
    elif err == "yes":
        plt.errorbar(data[1], data[2], xerr=xerr, yerr=yerr, fmt='o', color='blue', elinewidth=3)
    
    if 1 in method:
        plt.plot(X1, Y1, color='green')    

    if 2 in method:
        ellipse_plot_method2(data)

    if 3 in method:
        sqr_min_fit(data)

    plt.xlabel('$\Delta$ R.A. (")') 
    plt.ylabel('$\Delta$ Dec (")')    
    
    

print('S2')
compair_methods(data_S2)
plt.title("Sammenligning af metoder for S2")
plt.legend(["Sgr A*", "S2", "Metode 1", "Metode 2","Metode 3"], bbox_to_anchor=(1.3,1), loc='upper right')


print('S38')
compair_methods(data_S38)
plt.title("Sammenligning af metoder for S38")
plt.legend(["Sgr A*", "S38", "Metode 1", "Metode 2","Metode 3"], bbox_to_anchor=(1.3,1), loc='upper right')

print('S55')
compair_methods(data_S55) 
plt.title("Sammenligning af metoder for S55")
plt.legend(["Sgr A*", "S55", "Metode 1", "Metode 2","Metode 3"], bbox_to_anchor=(1.3,1), loc='upper right')

print('S62')
compair_methods(data_S62) 
plt.title("Sammenligning af metoder for S62")
plt.legend(["Sgr A*", "S62", "Metode 1", "Metode 2","Metode 3"], bbox_to_anchor=(1.3,1), loc='upper right')







    
    

























#---------------------------------------------------------------------------------------------------------
# ------------------- Use this to fint the xy-pane tilt ---------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
def ellipse_plot_method2_test(data, phi, plot='true'):
    arc = 2
    R = np.arange(0,arc*np.pi, 0.01)

    x = data[1]
    y = data[2]
    
    M, m, c_x, c_y, angle = fit_ellipse(x,y)
    
    xx = c_x + M*np.cos(R)*np.cos(phi) - m*np.sin(R)*np.sin(phi)
    yy = c_y + M*np.cos(R)*np.sin(phi) + m*np.sin(R)*np.cos(phi)
    
    
#    plt.plot(x,y, '*', color='blue')
    if plot == 'true':
        plt.plot(xx,yy, color='red')
    
    return xx, yy
 
    
fit = ellipse_fit_2(data_S2)
a = fit[0]
e = fit[1]
phi = fit[2]
c_x = fit[3]
c_y = fit[4]
b = a*np.sqrt(1-e**2)


plt.figure()
#ellipse_plot_method2_test(data_S2, 0.5*np.pi)
ellipse_plot_method2_test(data_S2, phi)
plt.plot(0,0,'o', color='black')
plt.plot([c_x-a*e*np.cos(phi),c_x+a*e*np.cos(phi)], [c_y-a*e*np.sin(phi), c_y+a*e*np.sin(phi)], '*', color='blue')
plt.plot(c_x, c_y, 'x')
plt.xlim([-0.4, 0.2])
plt.ylim([-0.2,0.2])


angle = m.atan2( (c_y), (c_x))
focal_points_x = [c_x-a*e*np.cos(angle),c_x+a*e*np.cos(angle)]
focal_points_y = [c_y-a*e*np.sin(angle), c_y+a*e*np.sin(angle)]

plt.figure()
ellipse_plot_method2_test(data_S2, angle)
plt.plot(0,0,'o', color='black')
plt.plot(focal_points_x, focal_points_y, '*', color='blue')
plt.plot(c_x, c_y, 'x')
plt.plot(c_x, c_y)
plt.xlim([-0.4, 0.2])
plt.ylim([-0.2,0.2])



#              !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! NEEDS WORK
#---------------------------------------------------------------------------------------------------------
# ------------------- Finding the yz-plane tilt (inclination) ---------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
inc = 0.5
#angle = 0.5*np.pi

xx = ellipse_plot_method2_test(data_S2, angle, plot='no')[0]
yy = ellipse_plot_method2_test(data_S2, angle, plot='no')[1]
zz = np.array([yy[i]*np.tan(inc) for i in range(len(xx))])

test_data = [[0], xx, yy]
r, theta = r_theta_data(test_data)

focal_data = [[0], focal_points_x, focal_points_y]
r_f, theta_f = r_theta_data(focal_data)

X = []
Y = []
for i in range(len(r)):
    X.append(r[i]*np.cos(theta[i] - angle + 0.5*np.pi))
    Y.append(r[i]*np.sin(theta[i] - angle + 0.5*np.pi))

X_f = []
Y_f = []
for i in range(len(r_f)):
    X_f.append(r_f[i]*np.cos(theta_f[i] - angle + 0.5*np.pi))
    Y_f.append(r_f[i]*np.sin(theta_f[i] - angle + 0.5*np.pi))

C_x = 0
C_y = min(Y_f) + a*e

plt.figure()
plt.plot(X, Y)
plt.plot(C_x, C_y, 'x')
plt.plot(0,0, 'o', color='black')
plt.plot(X_f, Y_f, '*', color='blue')
plt.xlim([-0.4, 0.2])
plt.ylim([-0.2,0.2])





fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot(xx,yy, zz, c='red')
ax.plot(xx,yy, c='blue')
ax.plot([0],[0],[0], 'o', c='black')
ax.view_init(azim=0, elev=90)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot(xx,yy, zz, c='red')
ax.plot(xx,yy, c='blue')
ax.plot([0],[0],[0], 'o', c='black')
plt.show()



#def f_inc(inc):
#    function_1 = C_y*np.sqrt(1+np.tan(inc))
#    function_2 = a/np.cos(inc) * np.sqrt(1-(b*np.cos(inc)/a)**2)
#    function = abs(function_1 - function_2)
#    return function 
#guess_init = [0.25*np.pi]
#bnds = (0,2*np.pi)
#result = opt.minimize(f_inc, guess_init, method='L-BFGS-B') #, bounds=bnds)    
#print(result.x)

#dist = []
#es = []
#index = []
#for i in np.linspace(-0.5*np.pi,0.5*np.pi, 20):
#    yy_test = np.linspace(C_y-a, C_y+a, 100)
#    zz_test = np.array([yy_test[j]*np.tan(i) for j in range(len(yy_test))])    
#    a_test = np.sqrt((yy_test[0] - yy_test[-1])**2 + (zz_test[0] - zz_test[-1])**2)*0.5
#    e_test = np.sqrt(1- (b/a_test)**2)
#    C_y_test = min(yy_test) + abs(yy_test[0]-yy_test[-1])/2
#    C_z_test = min(zz_test) + abs(zz_test[0]-zz_test[-1])/2
    
#    r_test = a_test*e_test
#    f_y_test = [C_y_test + r_test*np.cos(i), C_y_test - r_test*np.cos(i)]
#    f_z_test = [C_z_test + r_test*np.sin(i), C_z_test - r_test*np.sin(i)]
#    es.append(e_test)

#    dists = [np.sqrt( (f_y_test[j])**2 + (f_z_test[j])**2 ) for j in range(len(f_y_test))]   
#    dist.append(min(dists))
#    if dist[-1] == min(dist):
#        index = i
#print(dist)    
#    plt.figure()
#    plt.plot(yy_test,zz_test)
#    plt.plot(0,0,'o', color='black')
#    plt.plot(C_y_test, C_z_test, 'x')
#    plt.plot(f_y_test, f_z_test, '*')

#print(index)







#---------------------------------------------------------------------------------------------------------
# ------------------- My own minimized function 2 ---------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
def my_f_ellipse(a, e, phi, xf, yf, theta):
    r = a*(1-e**2)/(1-e*np.cos(theta - phi))
    x = r*np.cos(theta) + xf
    y = r*np.sin(theta) + yf
    return r, [x,y]


#a = 1
#e = 0.8
#phi = 0.25*np.pi
#xf = 10
#yf = 5
#for i in np.linspace(0,2*np.pi, 360):
#    x = my_f_ellipse(a, e, phi, xf, yf, i)[0]
#    y = my_f_ellipse(a, e, phi, xf, yf, i)[1]
#    plt.plot(x,y,'*')
#plt.plot(xf, yf, 'x')
#plt.xlim([-2*a+xf, 2*a+xf])
#plt.ylim([-2*a+yf,2*a+yf])
#plt.figure()

def residual2(a, e, phi, xf, yf, theta, r):
    return (r - my_f_ellipse(a,e,phi,xf,yf,theta)[0])**2
    

def my_ultimat_fit(data, compair='no'):
    
    def sqr_sum_f_ellipse(parameters):
        a, e, phi, xf, yf = parameters
        r, theta = r_theta_data(data, x_F = xf, y_F=yf)
        
        residuals = []
        
        for i in range(len(r)):
            R = residual2(a,e,phi,xf,yf,theta[i],r[i])
            R = R/(data[3][i]+data[4][i])
            residuals.append(R)
            
        return sum(residuals)
    
#    r, theta = r_theta_data(data)
#    furthest_point = r.index(max(r))
#    fp = furthest_point
        
#    phi_guess = theta[fp]
#    a_max = 2*max(r)
#    a_min = max(r)
        
#    center_area_x = (min(0,min(data[1])), max(0,max(data[1])))
#    center_area_y = (min(0,min(data[2])), max(0,max(data[2])))

    guess_init = [0.1, 0.5, 0.5*np.pi, 0, 0]
#    guess_init = [0.1, 0.8, 0.5*np.pi, 0, 0]
#    guess_init = [0.125, 0.88, 0.5*np.pi, -0.0151, 0.0686]
#    guess_init = ellipse_fit_2(data)
    
    bnds = ((0,0.5), (0.01,0.99), (-np.pi,np.pi), (-0.4,0.2), (-0.2,0.2))
#    bnds = ((a_min,a_max), (0.5,0.99), (phi_guess-0.25*np.pi, phi_guess + 0.25*np.pi), center_area_x, center_area_y)
    
    result = opt.minimize(sqr_sum_f_ellipse, guess_init, method='L-BFGS-B', bounds=bnds)    
    
    X1 = []
    Y1 = []
        
    a, e, phi, xf, yf = result.x
        
    for i in np.linspace(0, 2*np.pi, 720):
        theta = i 
        r = my_f_ellipse(a,e,phi,xf,yf,theta)[0]
        x = r*np.cos(i) + xf
        y = r*np.sin(i) + yf
            
        X1.append(x)
        Y1.append(y)
    
    if compair == 'no':
        plt.figure()
        plt.plot(xf, yf, 'x')
        plt.plot(xf+2*a*e*np.cos(phi), yf+2*a*e*np.sin(phi), 'x')
        plt.plot(data[1], data[2], '*')
        plt.plot(0,0,'o', color='black')

    plt.plot(X1, Y1, color='blue')

    
    return result.x, X1, Y1


def improve_data(data, error_limit=0, multiplyer=0.8):
    if error_limit==0:
        error_limit = (max(data[3])+max(data[4]))*multiplyer
        
    index_list = []
    for i in range(len(data[3])):
        if (data[3][i] + data[4][i]) < error_limit:
            index_list.append(i)
    
    a = []
    b = []
    c = []
    d = []
    e = []

    for i in index_list:
        a.append(data[0][i])
        b.append(data[1][i])
        c.append(data[2][i])
        d.append(data[3][i])
        e.append(data[4][i])
    
    data_improved = np.array([a,b,c,d,e])
    return data_improved



print(my_ultimat_fit(improve_data(data_S2)))
plt.xlim([0.2, -0.4])
plt.ylim([-0.2,0.2])
plt.title("Ultimate fit on improved S2 data")




















#---------------------------------------------------------------------------------------------------------
#------------------------------------- General relativitetsteori------------------------------------------
#---------------------------------------------------------------------------------------------------------
def perihelion_precession(a, e, P):
    P = P*365.25*24*60**2
    a = arcsec_to_pc(a)[0]
    
    epsilon = 24*np.pi**3 * a**2 / (P**2 * (3*10**8)**2 * (1-e**2))
    return epsilon, np.degrees(epsilon)


def perihelion_precession_data(data):
    P = period(data)
    a = plsq(data)[0][0]
    e = plsq(data)[0][1]
    
    epsilon_r, epsilon_d = perihelion_precession(a, e, P)
    return epsilon_r, epsilon_d
















































fit = ellipse_fit_2(improve_data(data_S2))
a = fit[0]
e = fit[1]
phi = fit[2]
c_x = fit[3]
c_y = fit[4]
b = a*np.sqrt(1-e**2)


plt.figure()
#ellipse_plot_method2_test(data_S2, 0.5*np.pi)
ellipse_plot_method2_test(improve_data(data_S2), phi)
plt.plot(0,0,'o', color='black')
plt.plot([c_x-a*e*np.cos(phi),c_x+a*e*np.cos(phi)], [c_y-a*e*np.sin(phi), c_y+a*e*np.sin(phi)], '*', color='blue')
plt.plot(c_x, c_y, 'x')
plt.xlim([-0.4, 0.2])
plt.ylim([-0.2,0.2])


angle = m.atan2( (c_y), (c_x))
focal_points_x = [c_x-a*e*np.cos(angle),c_x+a*e*np.cos(angle)]
focal_points_y = [c_y-a*e*np.sin(angle), c_y+a*e*np.sin(angle)]

plt.figure()
ellipse_plot_method2_test(improve_data(data_S2), angle)
plt.plot(0,0,'o', color='black')
plt.plot(focal_points_x, focal_points_y, '*', color='blue')
plt.plot(c_x, c_y, 'x')
plt.plot(c_x, c_y)
plt.xlim([-0.4, 0.2])
plt.ylim([-0.2,0.2])



#              !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! NEEDS WORK
#---------------------------------------------------------------------------------------------------------
# ------------------- Finding the yz-plane tilt (inclination) ---------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

#angle = 0.5*np.pi

xx = ellipse_plot_method2_test(improve_data(data_S2), angle, plot='no')[0]
yy = ellipse_plot_method2_test(improve_data(data_S2), angle, plot='no')[1]

test_data = [[0], xx, yy]
r, theta = r_theta_data(test_data)

focal_data = [[0], focal_points_x, focal_points_y]
r_f, theta_f = r_theta_data(focal_data)

X = []
Y = []
for i in range(len(r)):
    X.append(r[i]*np.cos(theta[i] - angle + 0.5*np.pi))
    Y.append(r[i]*np.sin(theta[i] - angle + 0.5*np.pi))

X_f = []
Y_f = []
for i in range(len(r_f)):
    X_f.append(r_f[i]*np.cos(theta_f[i] - angle + 0.5*np.pi))
    Y_f.append(r_f[i]*np.sin(theta_f[i] - angle + 0.5*np.pi))

C_x = 0
C_y = min(Y_f) + a*e

plt.figure()
plt.plot(X, Y)
plt.plot(C_x, C_y, 'x')
plt.plot(0,0, 'o', color='black')
plt.plot(X_f, Y_f, '*', color='blue')
plt.xlim([-0.4, 0.2])
plt.ylim([-0.2,0.2])




inc = np.arccos(np.sqrt( (b**2-min(Y_f)**2 - 2*min(Y_f)*np.sqrt(a**2-b**2))/b**2 ))
inc_answer = np.radians(133.818)

Z = np.array([Y[i]*np.tan(inc) for i in range(len(X))])
Z_answer = np.array([Y[i]*np.tan(inc_answer) for i in range(len(X))])

centerx, centery, centerz = C_x, C_y, C_y*np.tan(inc)
center_answerx, center_answery, center_answerz = C_x, C_y, C_y*np.tan(inc_answer)

endsx = [0,0]
endsy = [centery+a, centery-a]
endsz = [centerz+a*np.tan(inc), centerz-a*np.tan(inc)]
ends_answerx = [0,0]
ends_answery = [center_answery+a, center_answery-a]
ends_answerz = [center_answerz+a*np.tan(inc_answer), center_answerz-a*np.tan(inc_answer)]

focus_dist = np.sqrt( (a/np.cos(inc))**2 - (b)**2 )
focus_dist_answer = np.sqrt( (a/np.cos(inc_answer))**2 - (b)**2 )

focusx = [0,0]
focusy = [centery + focus_dist*np.cos(inc), centery - focus_dist*np.cos(inc)]
focusz = [centerz + focus_dist*np.sin(inc), centerz - focus_dist*np.sin(inc)]
focus_answerx = [0,0]
focus_answery = [center_answery + focus_dist_answer*np.cos(inc_answer), center_answery - focus_dist_answer*np.cos(inc_answer)]
focus_answerz = [center_answerz + focus_dist_answer*np.sin(inc_answer), center_answerz - focus_dist_answer*np.sin(inc_answer)]


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

ax.plot(X,Y, Z, c='red')
ax.plot(X,Y, c='blue')
ax.scatter(0,0,0, 'o', c='black')
ax.scatter(centerx, centery, centerz, 'o', color='blue')
ax.scatter(endsx, endsy, endsz, 'o', color='red')
ax.scatter(focusx, focusy, focusz, 'o', color='green')
ax.set_xlabel('x', fontsize = 20)
ax.set_ylabel('y', fontsize = 20)
ax.set_zlabel('z', fontsize = 20)
ax.set_xlim3d(-0.4, 0.2)
ax.set_ylim3d(-0.2,0.2)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

ax.plot(X,Y, Z_answer, c='red')
ax.plot(X,Y, c='blue')
ax.scatter(0,0,0, 'o', c='black')
ax.scatter(center_answerx, center_answery, center_answerz, 'o', color='blue')
ax.scatter(ends_answerx, ends_answery, ends_answerz, 'o', color='red')
ax.scatter(focus_answerx, focus_answery, focus_answerz, 'o', color='green')
ax.set_xlabel('x', fontsize = 20)
ax.set_ylabel('y', fontsize = 20)
ax.set_zlabel('z', fontsize = 20)
ax.set_xlim3d(-0.4, 0.2)
ax.set_ylim3d(-0.2,0.2)
plt.show()

























def f(t, a, e, omega, OMEGA, inc):
    A = (np.cos(omega)*np.cos(OMEGA) - np.sin(omega)*np.sin(OMEGA)*np.cos(inc)) #*a
    B = (np.cos(omega)*np.sin(OMEGA) + np.sin(omega)*np.cos(OMEGA)*np.cos(inc)) #*a
    F = (-np.sin(omega)*np.cos(OMEGA) - np.cos(omega)*np.sin(OMEGA)*np.cos(inc)) #*a
    G = (-np.sin(omega)*np.sin(OMEGA) + np.cos(omega)*np.cos(OMEGA)*np.cos(inc)) #*a
    
    r = my_f_ellipse(a,e,np.radians(0),0,0,t)[0]
    x = r*np.cos(t)
    y = r*np.sin(t)    
    
    X = A*x+F*y
    Y = B*x+G*y
    return X, Y

def the_mega_function(data,n,name):
    t = np.linspace(0,2*np.pi, 720)
    xdata = data[1]
    ydata = data[2]
    avg_dist_list = []
    a_list = []
    e_list = []
    omega_list = []
    OMEGA_list = []
    inc_list = []
    
    nummer = 0
    
    for a in np.linspace(0.02,0.2,n):
        for e in np.linspace(0.5,0.99,n):
            for omega in np.linspace(0,2*np.pi,n):
                for OMEGA in np.linspace(0,2*np.pi,n):
                    for inc in np.linspace(0,2*np.pi,n):
                        a_list.append(a)
                        e_list.append(e)
                        omega_list.append(omega)
                        OMEGA_list.append(OMEGA)
                        inc_list.append(inc)
                        
                        nummer = nummer + 1
                        print("We are now at:", nummer, "out of:", n**5, " ", round(nummer/n**5 * 100,2),"percent done, now on:", name)
                        X,Y = f(t, a, e, omega, OMEGA, inc)
                        
                        dist_all_points = []
                        for i in range(len(xdata)):
                            min_dist_to_point = 10
                            for j in range(len(X)):
                                d = np.sqrt( (xdata[i] - X[j])**2 + (ydata[i] - Y[j])**2 )#/(np.sqrt(data[3][i]**2+data[4][i]**2))
                                if d < min_dist_to_point:
                                    min_dist_to_point = d
                            dist_all_points.append(min_dist_to_point**2)
#                            dist_all_points.append(min_dist_to_point)
                        avg_dist = sum(dist_all_points)/len(dist_all_points)
                        avg_dist_list.append(avg_dist)
    
    index = avg_dist_list.index(min(avg_dist_list))
    return a_list[index], e_list[index], omega_list[index], OMEGA_list[index], inc_list[index], min(avg_dist_list)










# Method 4 ? 
def fit_method4(data, a0=0.1, e0=0.7, omega0=0, OMEGA0=0, inc0=0):
    
    def residual_4(parameters):
        a,e,omega,OMEGA,inc = parameters
        dist = []
        t = np.linspace(0,2*np.pi,720)
        X, Y = f(t,a,e,omega,OMEGA,inc)
        
        for j in range(len(data[1])):
            dist_min = 10**2
            for i in range(len(X)):
                distance = ((data[1][j] - X[i])**2 + (data[2][j] - Y[i])**2)#/(data[3][j]**2 + data[4][j]**2)
                if distance < dist_min:
                    dist_min = distance
            dist.append(dist_min)
                    
        return sum(dist)/len(dist)
    
    guess_init = [a0,e0,omega0,OMEGA0,inc0]
    
    bnds = ((0.05, 0.3), (0.3,0.99), (0,2*np.pi), (0,2*np.pi), (0,2*np.pi))
    
    result = opt.minimize(residual_4, guess_init, method='L-BFGS-B', bounds=bnds)  
    
    return result.x, result.fun





# ------------------------------------------------------------------------------------------------------------------
# ------------------------- Comparing 4 models ---------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def compair_methods(data, err = "no", method = [1,2,3,4], a0=0, e0=0, omega0=0, OMEGA0=0, inc0=0):
    X1 = []
    Y1 = []
    
    p = plsq(data)[0]
    S = r_theta_data(data)
    d = diff(S[0], S[1])
    
    for i in np.linspace(0, 2*np.pi, 720):
        theta = i + d
        r = f_ellipse(theta, p)
        x = r*np.cos(i)
        y = r*np.sin(i)
        
        X1.append(x)
        Y1.append(y)
        
    xerr = np.array(data[3])
    yerr = np.array(data[4])


    plt.figure()
    plt.plot(0,0,'o', color='black')
    
    if err == "no":
        plt.plot(data[1], data[2], '*', color='blue')
  
    elif err == "yes":
        plt.errorbar(data[1], data[2], xerr=xerr, yerr=yerr, fmt='o', color='blue', elinewidth=3)
    
    if 1 in method:
        plt.plot(X1, Y1, color='green')    

    if 2 in method:
        ellipse_plot_method2(data)

    if 3 in method:
        my_ultimat_fit(data, compair = 'yes')

    if 4 in method:
        t = np.linspace(0,2*np.pi, 720)
        x,y = f(t,a0,e0,omega0,OMEGA0,inc0)
        plt.plot(x,y, c='orange')

    plt.xlabel('$\Delta$ R.A. (")') 
    plt.ylabel('$\Delta$ Dec (")')    
    
    

print('S2')
compair_methods(data_S2, a0=0.11081081081081082, e0=0.7337142857142857, omega0=3.629847316397324, OMEGA0=4.549608598728532, inc0=7.086273817946454)
plt.title("Sammenligning af metoder for S2")
plt.legend(["Sgr A*", "S2", "Metode 1", "Metode 2","Metode 3","Metode 4"], bbox_to_anchor=(1.3,1), loc='upper right')


print('S38')
compair_methods(data_S38, a0=0.17500000000000004, e0=0.6247058823529412, omega0=0.7367963403774639, OMEGA0=2.9102745323056194, inc0=0.9398319447640624)
plt.title("Sammenligning af metoder for S38")
plt.legend(["Sgr A*", "S38", "Metode 1", "Metode 2","Metode 3", "Metode 4"], bbox_to_anchor=(1.3,1), loc='upper right')

print('S55')
compair_methods(data_S55, a0=0.10678134, e0=0.65623184, omega0=3.74771264, OMEGA0=2.23963716, inc0=2.49424632) 
plt.title("Sammenligning af metoder for S55")
plt.legend(["Sgr A*", "S55", "Metode 1", "Metode 2","Metode 3", "Metode 4"], bbox_to_anchor=(1.3,1), loc='upper right')

print('S62')
compair_methods(data_S62, a0=0.08626943, e0=0.98964251, omega0=2.38945621, OMEGA0=4.83138815, inc0=2.39082168) 
plt.title("Sammenligning af metoder for S62")
plt.legend(["Sgr A*", "S62", "Metode 1", "Metode 2","Metode 3", "Metode 4"], bbox_to_anchor=(1.3,1), loc='upper right')
plt.xlim([-0.12, 0.05])
plt.ylim([-0.025, 0.125])










#---------------------------------------------------------------------------------------------------------
# ------------------- How good are the fits? ---------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
def fit_precision2(data, f_t = 100, p_init = p0, test_interval = 1/8*np.pi):
    sq_dist_list = []
    data_x = data[1]
    data_y = data[2]
    
    fit = ellipse_plot_method2(data)
    fit_x = fit[0]
    fit_y = fit[1]
    
    for i in range(len(data_x)):
        dists = [np.sqrt( (data_x[i] - fit_x[j])**2 + (data_y[i] - fit_y[j])**2 ) for j in range(len(fit_x))]
        sq_dist_list.append(min(dists))
        
    return sum(sq_dist_list)/len(sq_dist_list)
        
        
    return sum(sq_dist_list)/len(data[0])

def fit_precision3AndAHalf(data, f_t = 100, p_init = p0, test_interval = 1/8*np.pi):
    sq_dist_list = []
    d = r_theta_data(data)
    fit = sqr_min_fit(data)
    p = [fit[0], fit[1]]
    phi = fit[2]
    
    for i in range(len(d[0])):
        sq_dist = []
        for x in np.linspace(d[1][i]-test_interval, d[1][i]+test_interval, f_t):
            sq_dist.append( np.sqrt((d[0][i]*np.sin(d[1][i]) - f_ellipse(x,p,phi)*np.sin(x))**2 + (d[0][i]*np.cos(d[1][i]) - f_ellipse(x,p,phi)*np.cos(x))**2) )
        sq_dist_list.append(min(sq_dist))
        
    return sum(sq_dist_list)/len(data[0])


def fit_precision3(data, f_t = 100, p_init = p0, test_interval = 1/8*np.pi):
    sq_dist_list = []
    data_x = data[1]
    data_y = data[2]
    
    a,e,phi,xf,yf = my_ultimat_fit(data)[0]

    X = []
    Y = []
    for i in np.linspace(0,2*np.pi, 1080):
        x,y = my_f_ellipse(a,e,phi,xf,yf,i)[1]
        X.append(x)
        Y.append(y)

    for i in range(len(data_x)):
        sq_dist = []
        for j in range(len(X)):
            sq_dist.append( np.sqrt((data_y[i] - Y[j])**2 + (data_x[i] - X[j])**2) )
        sq_dist_list.append(min(sq_dist))
        
    return sum(sq_dist_list)/len(data[0])#, X,Y, sq_dist_list


def fit_precision4(data, a=0, e=0, omega=0, OMEGA=0, inc=0):
    x = data[1]
    y = data[2]
    
    if a == 0:
        a,e,omega,OMEGA,inc = fit_method4(data)[0]
    
    t = np.linspace(0,2*np.pi, 1080)
    X,Y = f(t,a,e,omega,OMEGA,inc)

    dist_list = []
    for i in range(len(x)):
        dist = []
        for j in range(len(X)):
            dist.append( np.sqrt( (x[i]-X[j])**2 + (y[i]-Y[j])**2 ) )
        dist_list.append(min(dist))
    
    return sum(dist_list)/len(dist_list)

def fit_values(data, a=0, e=0, omega=0, OMEGA=0, inc=0):
    method1 = plsq(data)[0]
    method2 = ellipse_fit_2(data)
#    method3 = sqr_min_fit(data)
    method3 = my_ultimat_fit(data)[0]
    method4 = fit_method4(data,a,e,omega,OMEGA,inc)[0]
    
    goodness1 = fit_precision1(data)
    goodness2 = fit_precision2(data)
    goodness3 = fit_precision3(data)
    goodness4 = fit_precision4(data, a, e, omega, OMEGA, inc)
    return [method1[0], method1[1], goodness1], [method2[0], method2[1], goodness2], [method3[0], method3[1], goodness3], [method4[0], method4[1], goodness4]
    

print(fit_values(data_S2, a=0.11081081081081082, e=0.7337142857142857, omega=3.629847316397324, OMEGA=4.549608598728532, inc=7.086273817946454))
print("\n")
print(fit_values(data_S38, a=0.17500000000000004, e=0.6247058823529412, omega=0.7367963403774639, OMEGA=2.9102745323056194, inc=0.9398319447640624))
print("\n")
print(fit_values(data_S55, a=0.10678134, e=0.65623184, omega=3.74771264, OMEGA=2.23963716, inc=2.49424632))
print("\n")
print(fit_values(data_S62, a=0.08626943, e=0.98964251, omega=2.38945621, OMEGA=4.83138815, inc=2.39082168))

#X = fit_precision4(data_S2)[1]
#Y = fit_precision4(data_S2)[2]

#plt.figure()
#plt.plot(X,Y)
#plt.plot(0,0,'o')





X1, Y1 = my_ultimat_fit(improve_data(data_S2, multiplyer=0.5))[1], my_ultimat_fit(improve_data(data_S2, multiplyer=0.5))[2]
X2, Y2 = my_ultimat_fit(improve_data(data_S2, multiplyer=0.8))[1], my_ultimat_fit(improve_data(data_S2, multiplyer=0.8))[2]
plt.plot(X1,Y1)
plt.plot(X2,Y2)
plt.xlim([0.1,-0.25])
plt.ylim([-0.2, 0.2])

X = my_ultimat_fit(improve_data(data_S2, multiplyer=0.5))[1]
Y = my_ultimat_fit(improve_data(data_S2, multiplyer=0.5))[2]
list_of_dist = []
dists = []
for i in range(len(X)):
    for j in range(len(Y)):
        dist = np.sqrt( (X[i] - X[j])**2 + (Y[i] - Y[j])**2 )
        dists.append(np.sqrt(X[i]**2 + Y[i]**2))
        list_of_dist.append(dist)
        
print(max(list_of_dist))






# --------------------- Usikkerheder ------------------------------------------------------------------------------

a, e, phi, xf, yf = my_ultimat_fit(data_S2) # Method 4

X1 = []
Y1 = []
        
for i in np.linspace(0, 2*np.pi, 720):
    theta = i 
    r = my_f_ellipse(a,e,phi,xf,yf,theta)[0]
    x = r*np.cos(i) + xf
    y = r*np.sin(i) + yf
            
    X1.append(x)
    Y1.append(y)

X1, Y1 = ellipse_plot_method2(data_S2) # Method 2
X1, Y1 = sqr_min_fit(data_S2, XY="True") # Method 3

X = data_S2[1]
Y = data_S2[2]
X_err = data_S2[3]
Y_err = data_S2[4]

i_list = []
j_list = []

for i in range(len(X)):
    dist_min = 10
    for j in range(len(X1)):
        dist = np.sqrt( (X[i]-X1[j])**2 + (Y[i]-Y1[j])**2 ) 
        if dist < dist_min:
            dist_min = dist
            I = i
            J = j
    i_list.append(I)
    j_list.append(J)

fits = []
miss_fits = []

for i in range(len(i_list)):
    dist_x = abs(X[i_list[i]]-X1[j_list[i]])
    dist_y = abs(Y[i_list[i]]-Y1[j_list[i]])

    if dist_x <= X_err[i_list[i]] and dist_y <= Y_err[i_list[i]]:
        fits.append(i)
    else:
        miss_fits.append(i)
























# ---------------------------------------- TESTING SOME ODD STUFF -------------------------------------------------
def ellipse_fit_2(data):
    x = data[1]
    y = data[2]
    
    fit = fit_ellipse(x,y)
    
    a = fit[0]
    e = np.sqrt(1 - (fit[1]/fit[0])**2)
    phi = fit[-1]
    c_x = fit[2]
    c_y = fit[3]
    
    return a, e, phi, c_x, c_y



def ellipse_plot_method2(data):
    arc = 2
    R = np.arange(0,arc*np.pi, 0.01)

    x = data[1]
    y = data[2]
    
    M, m, c_x, c_y, phi = fit_ellipse(x,y)
    
    xx = c_x + M*np.cos(R)*np.cos(phi) - m*np.sin(R)*np.sin(phi)
    yy = c_y + M*np.cos(R)*np.sin(phi) + m*np.sin(R)*np.cos(phi)
    
    
#    plt.plot(x,y, '*', color='blue')
    plt.plot(xx,yy, color='red')
    
    return xx, yy


a, e, phi, c_x, c_y = ellipse_fit_2(improve_data(data_S2, multiplyer=0.9))
X = improve_data(data_S2, multiplyer=0.9)[1]
Y = improve_data(data_S2, multiplyer=0.9)[2]

plt.plot(X,Y, '*')
plt.plot(0,0,'o', color='black')
ellipse_plot_method2(improve_data(data_S2, multiplyer=0.9))
plt.plot([c_x+a*e*np.cos(phi),c_x-a*e*np.cos(phi)], [c_y+a*e*np.sin(phi),c_y-a*e*np.sin(phi)], '*', color='green')
plt.xlim([0.15, -0.3])
plt.ylim([-0.2,0.2])




ri, xi, yi = my_ultimat_fit(improve_data(data_S2))
r, x, y = my_ultimat_fit(data_S2)

plt.figure()
plt.title("LOOK AT THIS ONE")
plt.plot(x_S2, y_S2,'*')
plt.ylim([-0.05,0.2])
plt.xlim([-0.15,0.1])
plt.plot(x,y)
plt.plot(xi,yi)



# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
def obs_to_real(data, omega, OMEGA, inc): 
    X = data[1]
    Y = data[2]

    A = (np.cos(omega)*np.cos(OMEGA) - np.sin(omega)*np.sin(OMEGA)*np.cos(inc)) 
    C = (np.cos(omega)*np.sin(OMEGA) + np.sin(omega)*np.cos(OMEGA)*np.cos(inc)) 
    B = (-np.sin(omega)*np.cos(OMEGA) - np.cos(omega)*np.sin(OMEGA)*np.cos(inc)) 
    D = (-np.sin(omega)*np.sin(OMEGA) + np.cos(omega)*np.cos(OMEGA)*np.cos(inc))  
    
    x = []
    y = []
    for i in range(len(X)):
        x.append( (X[i]/A - Y[i]*B/(A*D))/(1-B*C/(A*D)) )
        y.append( (Y[i]/D - X[i]*C/(D*A))/(1-B*C/(A*D)) )
        
#        x.append(A*X[i]+B*Y[i])
#        y.append(C*X[i]+D*Y[i])
        
    return x,y


XX, YY = ellipse_plot_method2(data_S2)
data_S2_fit = [data_S2[0],XX,YY]

omega = 2.06
OMEGA = np.radians(227.85)
inc = np.radians(133.818)

x,y = obs_to_real(data_S2,omega,OMEGA,inc)

plt.plot(x,y,'*')
plt.plot(data_S2[1],data_S2[2],'*')


def f(t, a, e, omega, OMEGA, inc):
    A = (np.cos(omega)*np.cos(OMEGA) - np.sin(omega)*np.sin(OMEGA)*np.cos(inc)) #*a
    B = (np.cos(omega)*np.sin(OMEGA) + np.sin(omega)*np.cos(OMEGA)*np.cos(inc)) #*a
    F = (-np.sin(omega)*np.cos(OMEGA) - np.cos(omega)*np.sin(OMEGA)*np.cos(inc)) #*a
    G = (-np.sin(omega)*np.sin(OMEGA) + np.cos(omega)*np.cos(OMEGA)*np.cos(inc)) #*a
    
    r = my_f_ellipse(a,e,np.radians(0),0,0,t)[0]
    x = r*np.cos(t)
    y = r*np.sin(t)    
    
    X = A*x+F*y
    Y = B*x+G*y
    return X, Y



def multi_fit(data): 
    XX, YY = ellipse_plot_method2(data)

    a = 0.1254
    e = 0.88
    def residual(parameters):
        #a,e,omega,OMEGA,inc = parameters
        omega, OMEGA, inc = parameters
        dists = []
        t = np.linspace(0,2*np.pi,720)
        x,y = f(t,a,e,omega,OMEGA,inc)
        for i in range(len(XX)):
            dist_min = 10
            for j in range(len(x)):
                dist = np.sqrt( (XX[i] - x[i])**2 + (YY[i] - y[i])**2 )
                if dist < dist_min:
                    dist_min = dist
            dists.append(dist_min**2)
        avg_dist = sum(dists)/len(dists)
        return avg_dist

    guess_init = [0.11,0.88,2.06,np.radians(227),np.radians(133)]
    guess_init = [2.06,np.radians(227),np.radians(133)]
    
    bnds = [(0.01,1), (0.01,0.99), (0,2*np.pi), (0,2*np.pi), (0,2*np.pi)]
    bnds = [(0,2*np.pi), (0,2*np.pi), (0,2*np.pi)]
    
    result = opt.minimize(residual, guess_init, bounds = bnds)
    
    return result.x, result.fun


a4, e4, omega4, OMEGA4, inc4 = multi_fit(data_S2)[0]

t = np.linspace(0,2*np.pi,720)

X,Y = f(t,a4,e4,omega4,OMEGA4,inc4)

plt.figure()
plt.plot(X,Y,c='red')
plt.plot(XX,YY,c='green')






















def convert_to_real_ellipse(data, a, e, omega, OMEGA, inc):
    X = data[1]
    Y = data[2]
    A = (np.cos(omega)*np.cos(OMEGA) - np.sin(omega)*np.sin(OMEGA)*np.cos(inc)) 
    C = (np.cos(omega)*np.sin(OMEGA) + np.sin(omega)*np.cos(OMEGA)*np.cos(inc)) 
    B = (-np.sin(omega)*np.cos(OMEGA) - np.cos(omega)*np.sin(OMEGA)*np.cos(inc)) 
    D = (-np.sin(omega)*np.sin(OMEGA) + np.cos(omega)*np.cos(OMEGA)*np.cos(inc)) 
        
    y_list = []
    x_list = []
    
    for i in range(len(X)):
        y = (Y[i] - X[i]*C/A)/(D-B*C/A)
        x = (X[i] - B*y)/A
        y_list.append(y)
        x_list.append(x)

    return [data[0], x_list, y_list, data[3], data[4]]



def finding_areas(data, a, e, omega, OMEGA, inc):
    converted_data = convert_to_real_ellipse(data, a, e, omega, OMEGA, inc)
    r, theta = r_theta_data(converted_data)
    r_list = []
    
    for t in theta:        
        r = my_f_ellipse(a,e,np.radians(0),0,0,t)[0]
        r_list.append(r)

    areas = ellipse_area_ratios_ristricted(data, theta_min=np.radians(10), a=a, e=e, theta=theta, r=r_list)[1]        
    ratios = ellipse_area_ratios_ristricted(data, theta_min=np.radians(10), a=a, e=e, theta=theta, r=r_list)[0]
    avg_ratio = sum(ratios)/len(ratios)

    ratio_list = []
    for i in range(len(ratios)):
        ratio_list.appen( (ratios[i] - avg_ratio)**2 )
    
    return sum(ratio_list)/len(ratio_list), areas

# Now run the function above for all "best-fit" to find the very best fit (or most consistant that is)





