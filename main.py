#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 21:21:42 2021

@author: arnabghosh
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw,ImageFont
from scipy.fft import fft2,fftshift,fft,ifft
from scipy import stats
img_width = 64
img_height = 64
font_size = 70

'''Creating the image which will 
be used for creating the hologram'''
def create_image(text,img_width,img_height,font_size):  
    #Load the image and index over N/2 to N/2-1
    img = Image.new('1', (img_width, img_height), color='black')   
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',font_size)
    w,h = draw.textsize(text,font=font)
    #Draw text
    draw.text(((img_width-w)/2,(img_height-h)/2),text,font=font,fill=1)
    return(img)
#DFT or FFT of the complex array

'''Create the complex valued 
array with random phase to 
imitate a  diffused plane '''
string = '$'


img = create_image(string,img_width,img_height,font_size)


# img = Image.open('konoha.jpeg')
# img = img.convert('1')
# img = img.resize((64,64))
#Show Image
plt.figure(0)
plt.axis('off')
plt.imshow(img)
img = img.transpose(Image.TRANSPOSE)
img = img.rotate(0)
random_phase = np.random.rand(64,64)
random_phase = (random_phase-0.5)
random_phase = (random_phase)*2*np.pi
diffused_real = np.multiply(img,np.cos(random_phase))
diffused_im = np.multiply (img,np.sin(random_phase))
g = diffused_real + 1j*diffused_im

'''FFT of the 2d-image'''
F_g = fftshift(fft2((fftshift(g))))
max_mag = np.max(np.abs(F_g))
#print(max_mag)
F_real = np.real(F_g)
F_imag = np.imag(F_g)
'''Normalizing the maginitude and phase and Quantization'''
#print(F_imag)
#g_image = ifftshift(ifft2(g_fourier))
#print(np.max(np.abs(F_g)))'''
absF_g_norm = np.abs(F_g)/np.max(np.abs(F_g)) 
phaseF_g_norm = np.angle(F_g)/(2*np.pi)
#Fg_norm= np.multiply(absF_g_norm,np.exp(1j*np.angle(F_g)))
#print(Fg_norm)
'''Using a 8X8 cell and Quantizing the maginitude and phase'''
q_F_mag = np.floor(8*absF_g_norm+0.5)
q_F_phase = np.floor(8*phaseF_g_norm)
q_F = np.multiply(q_F_mag,np.exp(1j*q_F_phase*np.pi/4))
q_mag = np.zeros((img_width,img_height),dtype=float)
q_phase = np.zeros((img_width,img_height),dtype=float)
#error_real = absF_g_norm - q_F_mag
'''Error diffusion'''
e = 0+1j*0
for x in range(img_width):
    for y in range(img_height):
        real_new = F_real[x,y] - np.real(e)
        imag_new = F_imag[x,y] - np.imag(e)
        f = real_new + 1j*imag_new       
        #quantize f
        q_mag[x,y] = np.floor(8*(np.abs(f)/max_mag)+0.5)
        q_phase[x,y] = np.floor((8*(np.angle(f))/(2*np.pi)))
        q_real =((q_mag[x,y]/8)*max_mag) * np.cos(q_phase[x,y]*np.pi*0.25)
        q_imag =((q_mag[x,y]/8)*max_mag) * np.sin(q_phase[x,y]*np.pi*0.25)
        q_f = q_real + 1j*q_imag
        # error diffusion
        e = f - q_f 
        print(e)
print(np.max(q_phase))

        
'''Bitonal aperture'''

#aperture = np.zeros((8,8))

#final_image = np.zeros((512,512))
final_image = Image.new('1',(512,512),color='black')


for k in range(img_width):
    for l in range(img_height):
        aperture = Image.new('1',(8,8),color='black')
        
        pixmap = aperture.load()
        for y in range(int(q_F_mag[k,l])):
            #Lohmann Type 1
            pixmap[int(q_F_phase[k,l])+4,7-y] = 1
             
            #Lohmann Type 3
            
            # if(int(q_phase[k,l])==-4):
            #     pixmap[7,7-y]=1
            #     pixmap[1,7-y]=1
            #     pixmap[0,7-y]=1
            # elif(int(q_phase[k,l])==3):
            #     pixmap[0,7-y]=1
            #     pixmap[6,7-y]=1
            #     pixmap[7,7-y]=1
            # else:
            #     pixmap[int(q_phase[k,l])+4,(7-y)] = 1
            #     pixmap[int(q_phase[k,l])+3,(7-y)] = 1
            #     pixmap[int(q_phase[k,l])+5,(7-y)] = 1
        # plt.axis("off")
        # plt.imshow(aperture)
        # plt.show()
        # break
        final_image.paste(aperture,(8*k,8*l))
        
        
'''Finding out the hologram works or not'''        
inverse = np.log(np.abs((fftshift(fft2(fftshift(final_image)))))) 


"Plotting"

plt.imsave('hologram_dollar1.png',final_image,cmap='gray',dpi = 600)      
#Plot the image
plt.figure(1)
plt.axis('off')
#plt.hist(q_F_mag)
plt.imshow(inverse,cmap='gray')

'''Plotting the Magnitude distribution'''
plt.figure(2)

plt.hist(q_F_mag.flatten(),density = True , bins = 8)
plt.xlabel("Quantized Magnitude")
plt.ylabel("Likelihood")
plt.title("Rayleigh Distribution of the Magnitude values")
plt.show()

'''Plotting the Uniform Phase distribution'''
plt.figure(3)

plt.hist(q_F_phase.flatten(),density = True , bins = 8)
plt.xlabel("Quantized Phase")
plt.ylabel("Likelihood")
plt.title("Uniform Distribution of the Phase values")
plt.show()







