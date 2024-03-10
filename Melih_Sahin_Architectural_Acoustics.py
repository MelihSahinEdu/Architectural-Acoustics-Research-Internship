# Google Server Link: https://colab.research.google.com/drive/1TLhIFcfBY22xS6SxXSd1rPHcSadqzeOp?usp=sharing#scrollTo=EaVGKRDIRUvW
#Room Geometry Impulse Response Generator:
#Melih Şahin Koç University Summer 2022 Research Internship under the guiadnce of Professor Dr. Engin Erzin
#Except the part to get the audio file, and the definition to graph a sound file, all code completely belongs to the author, Melih Şahin.
#Average Running Time: 30 seconds on the provided Google Server


# the information below needs to be filled by the user
is_room_geometry_a_shoebox=True
room_dimensions=[18,19,49]; # x,y,z 
source=[3,9.5,5] # x,y,z
receiver=[16,10,13] # x,y,z
image_source_depth=1; #values between 1-8 are preferred, Exponential Time complexity=((number_of_surfaces-1)^image_source_depth)

number_of_surfaces=6
is_image_sources_inverse_law_open=True
is_air_attenuation_open=True #just for image sources, ray tracing air attenuatin is always open
custom_ray_number=True
our_ray_number=700 # Linear Time complexity=(Our_Ray_Number*number_of_frequency_bands)

# 10 frequency bands in total:

brick_absorpiton=[ 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.05, 0.05] #random absorption coefficient per band for construction material of brick
brick_scattering=[0.03, 0.05, 0.1, 0.2, 0.5, 0.6 ,0.6 ,0.7 ,0.8, 0.9]
fitting_constant=0.4
air_attenuation_coefficient=[ fitting_constant*0.1, fitting_constant*0.1, fitting_constant*0.1, fitting_constant*0.3, fitting_constant*0.6, fitting_constant*1, fitting_constant*1.9, fitting_constant*5.8, fitting_constant*20.3, fitting_constant*35] # 20 degrees, 40% humidity: compute energy loss throughout the air propagation, new_pressure=(old_pressure*(e^(-constant*x)))^(1/2), 0.05 is based on the frequecny distribution of popular music corresponding attenuation constant expectation

absorption_coefficient_of_surfaces=[brick_absorpiton,brick_absorpiton,brick_absorpiton,brick_absorpiton,brick_absorpiton,brick_absorpiton] # (+x,+y,-x,-y,+z,-z)
scatttering_coefficients_of_surfaces=[brick_scattering,brick_scattering,brick_scattering,brick_scattering,brick_scattering,brick_scattering] # (+x,+y,-x,-y,+z,-z)

number_of_frequency_bands=10 # this value is not to be changed
octave_bands_central_frequency=[31.3, 62.5, 125,250,500,1000,2000, 4000, 8000, 16000, 32000] # this array is not to be changed. length is 11 because of auchen page 72 f(n+1), central frequency
octave_bands=[[22,44],[44,88],[88,177],[177,355],[355,710],[720,1420],[1420,2840],[2840,5680],[5680,11360],[11360,22720]]


speed_of_sound=331.4 #meter per second
# (buradaki inverse law, sphere_radius, image_source_depth, number_of_races as a function, ... gibi şeyleri yapay öğrenme algoritmalarına optimize edilebilinirler

import cmath
import sys
import numpy
from IPython.display import Audio
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import scipy.signal
import librosa
import random
import math


room_volume=room_dimensions[0]*room_dimensions[1]*room_dimensions[2]
room_total_surface_area=2*room_dimensions[0]*room_dimensions[1]+2*room_dimensions[0]*room_dimensions[2]+2*room_dimensions[1]*room_dimensions[2]


# The function print_plot_play below and the code to get the adudio signal is excerpted from Prof. Dr. Engin Erzin's  Code
def print_plot_play(x, Fs, text=''):
    """
    1. Prints information on the audio signal
    2. Plots the signal waveform
    3. Creates player
    Args:
        x: Input signal
        Fs: Sampling rate of x
        text: Text to print
    """
    print('%s Fs = %d, x.shape = %s' % (text, Fs, x.shape))
    y = np.linspace(0, len(x) / Fs, num=len(x))
    plt.figure(figsize=(8, 2))
    plt.plot(y, x, color='gray')
    plt.xlabel('Time (sec)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    ipd.display(ipd.Audio(data=x, rate=Fs))


# Get the audio signal
!wget -O FurElise.ogg https://upload.wikimedia.org/wikipedia/commons/7/7b/FurElise.ogg
fn_ogg = 'FurElise.ogg'
music, Fs = librosa.load(fn_ogg, sr=None)
music=music - np.mean(music) # Mean remove
Fl = len(music)  # number of unit time




#dikdörtgenler prizmasının ve yüzeylerinin doğrusal cebirsel özellikleri:
#_6_yüzeyin_normal_vektörleri=[[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,1],[0,0,-1]]
_6_yüzeyin_normal_vektörleri=[[-1,0,0],[0,-1,0],[1,0,0],[0,1,0],[0,0,-1],[0,0,11]]
_6_yüzeyin_kaçıncı_koordinatları_sabit=[0,1,0,1,2,2]
_6_yüzeyin_sabit_kordinat_değerleri=[room_dimensions[0],room_dimensions[1],0,0,room_dimensions[2],0]
coordinates_of_any_points_of_surfaces=[[room_dimensions[0],0,0],[0,room_dimensions[1],0],[0,0,0],[0,0,0],[0,0,room_dimensions[2]],[0,0,0]]


#Useful Definitions ********************************************************************************************************************


def vector_scalar_product(vector,scalar):
    return [scalar*vector[0],scalar*vector[1],scalar*vector[2]]

def vector_addition(vector1,vector2):
    return [vector1[0]+vector2[0],vector1[1]+vector2[1],vector1[2]+vector2[2]]

def distance_between_two_vectors(vector1, vector2):
    sum=0
    for i in range(3):
        sum=sum+((vector2[i]-vector1[i])**2)
    return (sum**(1/2))

def normalize_the_vector(vector1):
    return vector_scalar_product(vector1,1/(((vector1[0]**2)+(vector1[1]**2)+(vector1[2]**2))**(1/2)))

def closest_time_on_a_line_to_a_point(point_on_line,line_vector, the_point):
    k,l,m,x,y,z,a,b,c=point_on_line[0],point_on_line[1],point_on_line[2],line_vector[0],line_vector[1],line_vector[2],the_point[0],the_point[1],the_point[2]
    time_scalar=(x*a-k*x+y*b-y*l+y*z-m*z)/((x**2)+(y**2)+(z**2))
    return time_scalar



def dot_product(vector1, vector2):
    product=0
    for a in range(3):
        product+=vector1[a]*vector2[a]
    return product

def cos_of_angle_between_two_vectors(vector1, vector2):
    return (dot_product(vector1,vector2))/((distance_between_two_vectors(vector1,[0,0,0]))*(distance_between_two_vectors(vector2,[0,0,0])))

def distance_vector_between_two_points(point1,point2):
    return [point2[0]-point1[0],point2[1]-point1[1],point2[2]-point1[2]]

def intersection_between_a_line_and_a_surface(line_point,line_vector,surface_number):
    if (dot_product(line_vector,_6_yüzeyin_normal_vektörleri[surface_number])==0): # if plane and the line do not intersect
        return -1
    negative_D=dot_product(_6_yüzeyin_normal_vektörleri[surface_number],coordinates_of_any_points_of_surfaces[surface_number])
    t=(negative_D-dot_product(_6_yüzeyin_normal_vektörleri[surface_number],line_point))/(dot_product(line_vector,_6_yüzeyin_normal_vektörleri[surface_number]))
    if t<0:
      return 0
    return (vector_addition(line_point,vector_scalar_product(line_vector,t)))

def intersection_between_final_image_and_final_surface(line_point,line_vector,surface_number):
    if (dot_product(line_vector,_6_yüzeyin_normal_vektörleri[surface_number])==0): # if plane and the line do not intersect
        return 0
    negative_D=dot_product(_6_yüzeyin_normal_vektörleri[surface_number],coordinates_of_any_points_of_surfaces[surface_number])
    t=(negative_D-dot_product(_6_yüzeyin_normal_vektörleri[surface_number],line_point))/(dot_product(line_vector,_6_yüzeyin_normal_vektörleri[surface_number]))
    return (vector_addition(line_point,vector_scalar_product(line_vector,t)))


def check_if_point_lies_on_a_surface(point,surface_number): # returns true if it in fact lies on that surface
    a=_6_yüzeyin_kaçıncı_koordinatları_sabit[surface_number]
    b=_6_yüzeyin_sabit_kordinat_değerleri[surface_number]
    if (abs(point[a]-b)<1e-300):
      if (a==0):
        if (point[1]>=0 and point[1]<=room_dimensions[1]):
          if (point[2]>=0 and point[2]<=room_dimensions[2]):
            return True

      elif (a==1):
        if (point[0]>=0 and point[0]<=room_dimensions[0]):
          if (point[2]>=0 and point[2]<=room_dimensions[2]):
            return True


      elif (a==2):
        if (point[0]>=0 and point[0]<=room_dimensions[0]):
          if (point[1]>=0 and point[1]<=room_dimensions[1]):
            return True


      return False


# 1.0 Image_Sources*********************************************************************************************************************************************************************************************




def reflection(vector, surface_number):
    normal_of_the_reflective_surface=_6_yüzeyin_normal_vektörleri[surface_number]
    dot_product=0
    for x in range(len(vector)):
        dot_product+=vector[x]*normal_of_the_reflective_surface[x]

    x1 = vector[0] - 2 * dot_product * normal_of_the_reflective_surface[0]
    y1 = vector[1] - 2 * dot_product * normal_of_the_reflective_surface[1]
    z1 = vector[2] - 2 * dot_product * normal_of_the_reflective_surface[2]

    return normalize_the_vector([x1,y1,z1])

def image_source_giver (point, surface_number):
  D=-dot_product(_6_yüzeyin_normal_vektörleri[surface_number],coordinates_of_any_points_of_surfaces[surface_number])
  distance_from_plane_to_point=abs(dot_product(_6_yüzeyin_normal_vektörleri[surface_number],point)+D)/distance_between_two_vectors(_6_yüzeyin_normal_vektörleri[surface_number],[0,0,0])
  if (dot_product(_6_yüzeyin_normal_vektörleri[surface_number],distance_vector_between_two_points(coordinates_of_any_points_of_surfaces[surface_number],point))>0): #point is in visibility of the normal of the surface
      image_source=vector_addition(point,vector_scalar_product(vector_scalar_product(_6_yüzeyin_normal_vektörleri[surface_number],-2),distance_from_plane_to_point))
  else:
      image_source=vector_addition(point,vector_scalar_product(vector_scalar_product(_6_yüzeyin_normal_vektörleri[surface_number],+2),distance_from_plane_to_point))
  return image_source

# 1.1 we form all the image source permutations
real_reflect_surface_orders=[]

def add(list):
    quantity = len(list)
    if quantity == 0:
        for d in range(number_of_surfaces):
            list.append([d])
    for b in range(quantity):
        for c in range(number_of_surfaces):
            if list[b][len(list[b]) - 1] != c:
                temporary = list[b].copy()
                temporary.append(c)
                list.append(temporary)


for a in range(image_source_depth):
    add(real_reflect_surface_orders)
# computation of all permutations in the list 'real_reflect_surface_orders' is done
o=len(real_reflect_surface_orders)
remaining_real_reflect_surface_orders=[]

all_reflections_with_ordered_coordinates=[]


for t in range(o):

    reflection_coordinates=[]
    reflection_coordinates.insert(0,receiver)
    image_source=source # just for defining the image_source
    for t1 in range(len(real_reflect_surface_orders[t])):
        image_source=image_source_giver(image_source,real_reflect_surface_orders[t][t1])


    vector_from_receiver_to_image_source=normalize_the_vector(vector_addition(image_source,vector_scalar_product(receiver,-1)))

    intersection_with_the_final_surface_coordinate=intersection_between_a_line_and_a_surface(receiver,vector_from_receiver_to_image_source,real_reflect_surface_orders[t][len(real_reflect_surface_orders[t])-1])

    if (intersection_with_the_final_surface_coordinate == 0 or intersection_with_the_final_surface_coordinate==-1 ):
        continue
    for o in range(3): # this is to compansete for computation slight errors due to the limitations of finite natura of computing
        if (abs(round(intersection_with_the_final_surface_coordinate[o])-intersection_with_the_final_surface_coordinate[o])<1e-10):
            intersection_with_the_final_surface_coordinate[o]=round(intersection_with_the_final_surface_coordinate[o])

    if (check_if_point_lies_on_a_surface(intersection_with_the_final_surface_coordinate,real_reflect_surface_orders[t][len(real_reflect_surface_orders[t])-1])):
        reflection_coordinates.insert(0,intersection_with_the_final_surface_coordinate)







        current_coordinate=receiver
        current_vector_direction=normalize_the_vector(distance_vector_between_two_points(receiver,intersection_with_the_final_surface_coordinate)) #gerçek yönün terinsi alıyoruz

        kontrol = 1
        for u3 in range(len(real_reflect_surface_orders[t])+1): # +1 is because of the fact that we also need to check between the source and the first surface


             current_surface=real_reflect_surface_orders[t][len(real_reflect_surface_orders[t])-1-u3]
             number=len(_6_yüzeyin_normal_vektörleri)
             closest_intersection_distance=0
             closest_surface=-1
             daha_clos_int_dis_değişti_mi=False

             for u4 in range(number):

                   distance_coordinate= intersection_between_a_line_and_a_surface(current_coordinate,current_vector_direction,u4)
                   if (distance_coordinate!=0 and distance_coordinate!=-1): # check if the time is unzero
                       distance = distance_between_two_vectors(distance_coordinate, current_coordinate)
                       if (distance != 0):
                           if not daha_clos_int_dis_değişti_mi:
                               closest_intersection_distance = distance
                               closest_surface=u4
                               daha_clos_int_dis_değişti_mi=True

                           else:
                               if (distance<closest_intersection_distance):
                                   closest_intersection_distance=distance
                                   closest_surface=u4

             if (u3==len(real_reflect_surface_orders[t])): #son kontrol
                 first_distance=distance_between_two_vectors(source,current_coordinate)
                 if (closest_intersection_distance<first_distance):



                     kontrol=0
                     break # this break is useless

             else:
                 if (closest_surface!=current_surface):

                     kontrol=0
                     break
                 else:

                     current_coordinate =reflection_coordinates[0]
                     current_vector_direction = reflection(current_vector_direction, current_surface)
                     if u3!=(len(real_reflect_surface_orders[t])-1):
                         reflection_coordinates.insert(0, intersection_between_a_line_and_a_surface(current_coordinate,current_vector_direction,real_reflect_surface_orders[t][len(real_reflect_surface_orders[t])-1-(u3+1)]))


        if (kontrol==1):

            reflection_coordinates.pop()
            all_reflections_with_ordered_coordinates.append(reflection_coordinates)
            remaining_real_reflect_surface_orders.append(real_reflect_surface_orders[t])










#print(remaining_real_reflect_surface_orders)
#print(all_reflections_with_ordered_coordinates)
#IMPORTANT: each element of  (all_reflections_with_ordered_coordinates) is without the coordiantes of the receiver and the source

def attenuation_loss_due_to_air_travel(pressure_value,path_length, band_no): # band_no is an element of {0,1,...,7}
    euler=2.718281828459045
    value=(euler**(-path_length*air_attenuation_coefficient[band_no]))**(1/2)

    return pressure_value*value

def attenuation_loss_due_to_air_travel_energy(pressure_value,path_length, band_no): # band_no is an element of {0,1,...,7}
    euler=2.718281828459045
    value=(euler**(-path_length*air_attenuation_coefficient[band_no]))

    return pressure_value*value


#we now compute the reverberation time according to the formula page 60 [] to obtain maximum time in the impulse response
reverberation_time=(0.16)*room_volume/room_total_surface_area
if reverberation_time<1:
  reverberation_time=1
max_time_in_seconds=reverberation_time
print(max_time_in_seconds)
#*************************************************************************

impulse_responses_single_band=[] #will include 8 band-specific impulse responses



max_time_sampling_unit=0
pressure_and_time_duration_of_each_path=[]
if image_source_depth==0:
  remaining_real_reflect_surface_orders=[]


all_dirac_deltas=[]



for band in range(number_of_frequency_bands):


    dirac_deltas =dirac_deltas_main = np.zeros((math.ceil(Fs * max_time_in_seconds) + 1,), dtype="float64")
    impulse_response = np.zeros((math.ceil(Fs * max_time_in_seconds) + 1,), dtype="float64")

    current_band_frequency=octave_bands_central_frequency[band]

    for i1 in range(len(remaining_real_reflect_surface_orders)):
        total_path_length = 0
        pressure = 1
        for y in range(len(remaining_real_reflect_surface_orders[i1])):

            if y==0:
                o=distance_between_two_vectors(source,all_reflections_with_ordered_coordinates[i1][y])
                total_path_length+=o

                pressure=pressure*(1-(absorption_coefficient_of_surfaces[remaining_real_reflect_surface_orders[i1][y]][band]))**(1/1)
                pressure=pressure*(1-(scatttering_coefficients_of_surfaces[remaining_real_reflect_surface_orders[i1][y]][band]))**(1/1)


            else:
                o=distance_between_two_vectors(all_reflections_with_ordered_coordinates[i1][y-1],all_reflections_with_ordered_coordinates[i1][y])
                total_path_length+=o

                pressure=pressure*(1-(absorption_coefficient_of_surfaces[remaining_real_reflect_surface_orders[i1][y]][band]))**(1/1)
                pressure=pressure*(1-(scatttering_coefficients_of_surfaces[remaining_real_reflect_surface_orders[i1][y]][band]))**(1/1)
            o=distance_between_two_vectors(all_reflections_with_ordered_coordinates[i1][len(all_reflections_with_ordered_coordinates[i1])-1],receiver)
            total_path_length+=o

        real_time=(total_path_length/speed_of_sound)

        unit_complex=z = complex(0,1)
        #the_phase_coefficient=(cmath.exp(z*2*math.pi*current_band_frequency*real_time)).real
        the_phase_coefficient=1
        pressure*=the_phase_coefficient

        first_discrete_location=math.ceil(real_time*Fs)

        if is_image_sources_inverse_law_open:
            pressure=pressure/((4*math.pi)*total_path_length)**2

        if is_air_attenuation_open:
              pressure=attenuation_loss_due_to_air_travel_energy(pressure,total_path_length,band)


        if (first_discrete_location)<= max_time_in_seconds*Fs:
            impulse_response[first_discrete_location]+=pressure
            the_phase_coefficient = (cmath.exp(z * 2 * math.pi * current_band_frequency * real_time)).real
            if the_phase_coefficient>0:
                dirac_deltas[first_discrete_location] = 1
            if the_phase_coefficient<0:
                dirac_deltas[first_discrete_location] = -1



        if max_time_sampling_unit<real_time:
            max_time_sampling_unit=real_time







    # in this part below the direct sound is added  ********************
    total_path_length=distance_between_two_vectors(receiver,source)
    pressure=1

    real_time=(total_path_length/speed_of_sound)

    unit_complex=z = complex(0,1)
    #the_phase_coefficient=(cmath.exp(z*2*math.pi*current_band_frequency*real_time)).real
    the_phase_coefficient=1
    pressure*=the_phase_coefficient
    first_discrete_location=math.ceil(real_time*Fs)

    if is_image_sources_inverse_law_open:
        pressure=pressure/((4*math.pi)*total_path_length)**2

    if is_air_attenuation_open:
        pressure=attenuation_loss_due_to_air_travel_energy(pressure,total_path_length,band)

    if (first_discrete_location)<= max_time_in_seconds*Fs:
        impulse_response[first_discrete_location]+=pressure
        the_phase_coefficient = (cmath.exp(z * 2 * math.pi * current_band_frequency * real_time)).real
        if the_phase_coefficient>0:
            dirac_deltas[first_discrete_location] = 1
        if the_phase_coefficient<0:
            dirac_deltas[first_discrete_location] = -1





    if max_time_sampling_unit<real_time:
        max_time_sampling_unit=real_time
    if image_source_depth!=0:
        impulse_responses_single_band.append(impulse_response)
        all_dirac_deltas.append(dirac_deltas)


#*********************************************************************

# form dirac deltas for each frequecny domain





# Image Sources'ın Bitişi ****************************************************************************************************************************

#RAY TRACING******************************************************************************************************************************
#yukarıda image sources'ı hesapladık şimdi ray tracing kısmına geçiyoruz
ray_in_sphere_number=0

ray_max_time=max_time_in_seconds
ray_band_impulse_responses=[]


radius_of_receiver_sphere=0.00001 #nurayı değiştir
room_volume=room_dimensions[0]*room_dimensions[1]*room_dimensions[2]

number_of_rays=3745*round(room_volume)
if (custom_ray_number):
    number_of_rays=our_ray_number


print(number_of_rays)


direction_of_rays=[]

# we create a uniformly distributed normal vector space according to   https://mathworld.wolfram.com/SpherePointPicking.html
def random_normal_vector_giver():
    x0,x1,x2,x3=random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1)
    while not ((x0**2+x1**2+x2**2+x3**2)>=1):
        x0,x1,x2,x3=random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1)

    h=((x0**2)+(x1**2)+(x2**2)+(x3**2))
    if (h==0):
        return [1,0,0] # since h=0 is of exremely low probability
    x=(2*(x1*x3+x0*x2))/h
    y=(2*(x2*x3-x0*x1))/h
    z=((x0**2)+(x3**2)-(x1**2)-(x2**2))/h
    if [x,y,z]!=[0,0,0]:
        return [x,y,z]


for u in range(number_of_rays): #burada def'i kullanmıyoruz çünkü süreci yavaşlatıyor, def random_normal_vector_giver'ı aşağıda farklı bir şekilde kullanacağız
    x0,x1,x2,x3=random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1)
    while not ((x0**2+x1**2+x2**2+x3**2)>=1):
        x0,x1,x2,x3=random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1)

    h=((x0**2)+(x1**2)+(x2**2)+(x3**2))
    if (h==0):
        continue
    x=(2*(x1*x3+x0*x2))/h
    y=(2*(x2*x3-x0*x1))/h
    z=((x0**2)+(x3**2)-(x1**2)-(x2**2))/h
    if [x,y,z]!=[0,0,0]:
        direction_of_rays.append([x,y,z])

#direction_of_rays=[[1,0,0]] # burayı çıkar
yüzey_sayısı=len(_6_yüzeyin_normal_vektörleri)


direction_vector=vector_addition(receiver,vector_scalar_product(source,-1)) # vector between the source and the receiver
direction_vector1=[direction_vector[0],direction_vector[1],0]
cos=(dot_product(direction_vector,direction_vector1))/((distance_between_two_vectors(direction_vector,[0,0,0]))*(distance_between_two_vectors(direction_vector,[0,0,0])))
pressure_of_each_ray=(2*1)/(number_of_rays*((distance_between_two_vectors(receiver,source))**2)*(1-cos))

for band1 in range(number_of_frequency_bands):
    print(band1)
    size=math.ceil(Fs * ray_max_time) + 1
    ray_impulse_response = np.zeros((size,), dtype="float64")

    for q in range(len(direction_of_rays)):


        current_ray_pressure=pressure_of_each_ray
        vector_time=0
        vector_current_direction=direction_of_rays[q]
        ray_current_position=source
        reflection1=0
        previous_reflection_surface_number=-1
        while reflection1<25 and vector_time<ray_max_time: #20'yi tamamen kendin seçtin değiştirebilirsin veya yerine bir zaman sınırı da koyabilirsin


            perpendecilar_scalar_time_between_the_sphere_and_the_ray=closest_time_on_a_line_to_a_point(ray_current_position,vector_current_direction,receiver)


            if (perpendecilar_scalar_time_between_the_sphere_and_the_ray>=0):

                covered_vector=vector_scalar_product(vector_current_direction,perpendecilar_scalar_time_between_the_sphere_and_the_ray)
                closest_point_on_the_ray_to_the_receiver=vector_addition(ray_current_position,covered_vector)
                r=distance_between_two_vectors(closest_point_on_the_ray_to_the_receiver,receiver)


                if (r<=radius_of_receiver_sphere):
                    distance_to_sphere_from_ray_origin=distance_between_two_vectors(ray_current_position,vector_addition(ray_current_position,vector_scalar_product(vector_current_direction,perpendecilar_scalar_time_between_the_sphere_and_the_ray)))
                    if is_air_attenuation_open:
                        current_ray_pressure=attenuation_loss_due_to_air_travel_energy(current_ray_pressure,distance_to_sphere_from_ray_origin,band1)
                    if (round(Fs*(vector_time+distance_to_sphere_from_ray_origin/speed_of_sound))<size):
                        ray_impulse_response[round(Fs*(vector_time+distance_to_sphere_from_ray_origin/speed_of_sound))]=(current_ray_pressure+ray_impulse_response[round(Fs*(vector_time+distance_to_sphere_from_ray_origin/speed_of_sound))]) #/2'yi sildim
                        ray_in_sphere_number+=1

                        break

            # surface variables to be processed for each surface
            most_recent_reflected_surface_number=0
            shortest_surface_time=0
            ilki_oldu_mu=False
            x,y,z=0,0,0
            for n in range(yüzey_sayısı):
                if n!=previous_reflection_surface_number:

                    orthogonality=dot_product(_6_yüzeyin_normal_vektörleri[n],vector_current_direction)

                    if (orthogonality!=0): # if surface plane and ray line are parallel we do not check for intersection

                        #we now compute the intersection points of the ray line and the surface plane:
                        t=(dot_product(_6_yüzeyin_normal_vektörleri[n],coordinates_of_any_points_of_surfaces[n])-dot_product(_6_yüzeyin_normal_vektörleri[n],ray_current_position))/orthogonality # we find the time when the ray line and the plane intersects




                        if (t>=0 and ((not ilki_oldu_mu) or t<shortest_surface_time)):
                            shortest_surface_time=t
                            ilki_oldu_mu=True
                            x=ray_current_position[0]+t*vector_current_direction[0]
                            y=ray_current_position[1]+t*vector_current_direction[1]
                            z=ray_current_position[2]+t*vector_current_direction[2]

                            most_recent_reflected_surface_number=n

            distance=distance_between_two_vectors(ray_current_position,[x,y,z])
            previous_reflection_surface_number=most_recent_reflected_surface_number
            if is_air_attenuation_open:
                current_ray_pressure=attenuation_loss_due_to_air_travel_energy(current_ray_pressure,distance,band1 )
            current_ray_pressure=current_ray_pressure*(1-absorption_coefficient_of_surfaces[most_recent_reflected_surface_number][band1])
            vector_time+=distance/speed_of_sound
            reflection1+=1

            s=scatttering_coefficients_of_surfaces[most_recent_reflected_surface_number][band1]
            vector_specular_direction=vector_scalar_product(reflection(vector_current_direction,most_recent_reflected_surface_number),(1-s))

            vector_random_scattered_direction=random_normal_vector_giver()
            yy=dot_product(_6_yüzeyin_normal_vektörleri[most_recent_reflected_surface_number],vector_random_scattered_direction)
            while (yy)<0:
                vector_random_scattered_direction=random_normal_vector_giver()
                yy=dot_product(_6_yüzeyin_normal_vektörleri[most_recent_reflected_surface_number],vector_random_scattered_direction)
            vector_random_scattered_direction=vector_scalar_product(vector_random_scattered_direction, s)
            final_direction_of_the_vector=vector_addition(vector_random_scattered_direction,vector_specular_direction)
            final_direction_of_the_vector=normalize_the_vector(final_direction_of_the_vector)

            vector_current_direction=final_direction_of_the_vector

            ray_current_position=[x,y,z]

            # now we apply the diffuse rain method *******
            dis=distance_between_two_vectors(ray_current_position,receiver)
            cos_of_angle1=cos_of_angle_between_two_vectors(distance_vector_between_two_points(ray_current_position,receiver),_6_yüzeyin_normal_vektörleri[most_recent_reflected_surface_number])
            #print(math.acos(cos_of_angle1)) # büyük sıkıntı var
            cos_of_angle2=cos_of_angle_between_two_vectors(distance_vector_between_two_points(ray_current_position,receiver),vector_specular_direction)

            constant1=2*cos_of_angle1 #tekrar gözden geçir - çıkıyor
            constant2=(1-math.cos(math.acos(cos_of_angle1)/2))
            energy_of_ray_diffusion=current_ray_pressure*constant1*constant2*scatttering_coefficients_of_surfaces[most_recent_reflected_surface_number][band1]
            y2=float(energy_of_ray_diffusion)
            energy_of_ray_diffusion=attenuation_loss_due_to_air_travel_energy(energy_of_ray_diffusion,dis,band1)

            if (round(Fs*(vector_time+dis/speed_of_sound))<size):
                if (energy_of_ray_diffusion<0):
                    energy_of_ray_diffusion*=-1
                if  (energy_of_ray_diffusion > 1.0e-200) :
                    ray_impulse_response[round(Fs*(vector_time+dis/speed_of_sound))]+=energy_of_ray_diffusion
                    current_ray_pressure-=energy_of_ray_diffusion
                    #ray_diffuse+=1
                    #print(ray_diffuse)
    if image_source_depth!=0:
        ray_impulse_response+=impulse_responses_single_band[band1]
    ray_band_impulse_responses.append(ray_impulse_response)

#Ray tracing'İn bitişi, sounaçta 10 tane energy time histogram elde ettik, şimdi bunları tekbir broadband pressure impulse response'a dönüştüreceğiz***********************************************************************************************************

# we will now do multiband filtering to obtain a broadband impulse response out of 10 band-specific ray energy histograms. This is done in accordance with the pages 71-72 [1]

# Generate Poisson-Distributed Dirac Deltas***********************************************************************************************************


t0 = 0.0014 * ((room_volume) ** (1 / 3))
pi = 3.1415926535897
constant1 = (4 * pi * (speed_of_sound ** 3)) / room_volume


def next_time(current_time):
    epsilon = 0.00000000001
    z = random.uniform(0, 1)
    delta = 0
    if (z <= (1 / 2)):
        delta = 1
    else:
        delta = -1

    z += +epsilon
    mean = constant1 * (current_time ** 2)
    difference = abs((1 / mean) * math.log(1 / mean))
    if difference < 1 / (Fs):
        difference = 1 / (Fs)

    return current_time + difference, delta


maximum_length = math.ceil(Fs*ray_max_time)
values = np.zeros(maximum_length, dtype="float64") #values is the dirac delta function

current_length = t0

delta = 1
the_one = round(current_length * Fs)

kontrol=False
while (the_one < maximum_length):
    # print(current_length)
    if kontrol:
        values[the_one] = delta

    time, delta = next_time(current_length)
    current_length = time
    the_one = round(current_length * Fs)
    kontrol=True


dirac_deltas_of_ray=[]
for qq in range(number_of_frequency_bands):
    q=np.copy(values)
    if image_source_depth!=0:
        for yy in range(len(values)):
            if all_dirac_deltas[qq][yy]!=0:
                q[yy]=all_dirac_deltas[qq][yy]






    frequncy_domain_of_dirac_deltas=np.fft.fft(q, len(values))
    #************ a new approach
    for h in range(len(frequncy_domain_of_dirac_deltas)):
        if h==0:
            pass
        elif h<=math.floor(len(frequncy_domain_of_dirac_deltas)/2):
            frequncy_domain_of_dirac_deltas[h]*=2
        else:
            frequncy_domain_of_dirac_deltas[h]=0

    #************
    dirac_deltas_of_ray.append(frequncy_domain_of_dirac_deltas)



filtered_ones = []

# step 3 we filter them according to the fore front of page 72 [2]
for r in range(number_of_frequency_bands):
    buffer = np.copy(dirac_deltas_of_ray[r])

    for h in range(len(buffer)):

        if h >= octave_bands[r][0] and h < octave_bands[r][1]:

            if h <= octave_bands_central_frequency[r]:

                buffer[h] = (1 / 2) * (1 + np.cos((2 * math.pi) * buffer[h] / octave_bands_central_frequency[r]))





            else:
                buffer[h] = (1 / 2) * (1 - np.cos((2 * math.pi) * buffer[h] / octave_bands_central_frequency[r + 1]))

        else:
            buffer[h] = 0
    # print_plot_play(x=np.abs(buffer), Fs=1, text=' this is he buffer : ' )
    filtered_ones.append(buffer)

# now we transform them into their time domain
sequences_in_time_domain = []
for r1 in range(number_of_frequency_bands):
    aa = np.fft.ifft(filtered_ones[r1], len(filtered_ones[r1])).real

    sequences_in_time_domain.append(aa)
    # print_plot_play(x=np.abs(filtered_ones[r1]), Fs=Fs, text=' a1 : ')
    # print_plot_play(x=aa, Fs=Fs, text=' a2 : ')

# we weight each of them with band-specific ray energy histogram
weighted_ones = []
for o1 in range(number_of_frequency_bands):
    aaa = np.zeros(len(frequncy_domain_of_dirac_deltas), dtype="float64") #aaa = np.zeros(round(len(frequncy_domain_of_dirac_deltas) / 2), dtype="float64")
    for o2 in range(len(frequncy_domain_of_dirac_deltas)):
        if o2==0:
            aaa[o2]=sequences_in_time_domain[o1][o2]*(((ray_band_impulse_responses[o1][o2])/(sequences_in_time_domain[o1][o2]**2))**(1/2))*(((octave_bands[o1][1]-octave_bands[o1][0])/(Fs/2))**(1/2))
        else:
            aaa[o2]=sequences_in_time_domain[o1][o2]*((((ray_band_impulse_responses[o1][o2])/(sequences_in_time_domain[o1][o2]**2+sequences_in_time_domain[o1][o2-1]**2))**(1/2))*(((octave_bands[o1][1]-octave_bands[o1][0])/(Fs/2))**(1/2)))

    weighted_ones.append(aaa)

# we now sum all ray band impulse responses
final_filtered_ray_impulse_response = np.zeros(len(frequncy_domain_of_dirac_deltas), dtype="float64")
for o3 in range(number_of_frequency_bands):
    final_filtered_ray_impulse_response+=weighted_ones[o3]

#print_plot_play(x=final_impulse_response, Fs=Fs, text=' image_sources_impulse_response : ')
print_plot_play(x=final_filtered_ray_impulse_response, Fs=Fs, text=' final_filtered_ray_impulse_response : ')



# *******************************************************


filtered_music = scipy.signal.convolve(final_filtered_ray_impulse_response[:,]/sum(final_filtered_ray_impulse_response[:,]), music[:,], mode='full')


print_plot_play(x=music[Fs*10:Fs*60], Fs=Fs, text=' original music : ' )
print_plot_play(x=filtered_music[Fs*10:Fs*60], Fs=Fs, text=' convoluted music : ')


#References:
#[1] Savioja L, Svensson UP. Overview of geometrical room acoustic modeling techniques. J Acoust Soc Am. 2015 Aug;138(2):708-30. doi: 10.1121/1.4926438. PMID: 26328688.
#[2] Dirk Schröder, Physically Based Real-Time Auralization of Interactive Virtual Environments, 2011, Thesis, RWTH Aachen
#[3] J. B. Allen and D. A. Berkley, “Image method for efficiently simulating small-room acoustics,” J. Acoust. Soc. Am. 65(4), 943–950 ,1979
#[4] Michael Vorländer, Auralization: Fundamentals of Acoustics, Modelling, Simulation, Algorithms and Acoustic Virtual Reality, 2nd edition, 2020
#[5] Vorländer M. Computer simulations in room acoustics: concepts and uncertainties. J Acoust Soc Am. 2013 Mar;133(3):1203-13. doi: 10.1121/1.4788978. PMID: 23463991
#[6] https://mathworld.wolfram.com/SpherePointPicking.html
