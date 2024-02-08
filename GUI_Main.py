
from tkinter import *
import tkinter as tk


from PIL import Image ,ImageTk

from tkinter.ttk import *
#from pymsgbox import *


root=tk.Tk()

root.title("Leukamia Disease Detection System")
w,h = root.winfo_screenwidth(),root.winfo_screenheight()

image2 =Image.open("gui.jpg")
image2 =image2.resize((w,h), Image.ANTIALIAS)

background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=20)
#, relwidth=1, relheight=1)

w = tk.Label(root, text="Leukamia Disease Detection System",width=75,background="orange",height=2,font=("Times new roman",25,"bold"))
w.place(x=0,y=0)



w,h = root.winfo_screenwidth(),root.winfo_screenheight()
root.geometry("%dx%d+0+0"%(w,h))
root.configure(background="skyblue")


from tkinter import messagebox as ms


def Login():
    from subprocess import call
    call(["python","Login.py"])
def Register():
    from subprocess import call
    call(["python","Registration.py"])


wlcm=tk.Label(root,text="...... Welcome to Leukamia Disease Detection System ......",width=85,height=1,background="orange",foreground="black",font=("Times new roman",22,"bold"))
wlcm.place(x=0,y=655)




d2=tk.Button(root,text="Login",command=Login,width=9,height=1,bd=0,background="grey",foreground="black",font=("times new roman",25,"bold"))
d2.place(x=10,y=10)


d3=tk.Button(root,text="Register",command=Register,width=9,height=1,bd=0,background="grey",foreground="black",font=("times new roman",25,"bold"))
d3.place(x=160,y=10)



root.mainloop()
