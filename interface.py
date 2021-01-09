# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 11:41:54 2020

@author: ntruo
"""



from tkinter import *
from tkinter.ttk import Frame, Button, Style

import model as ANLP

window = Tk()

models = ANLP.Model()

def clicked():
    review = txt_input.get("1.0","end")
    print(review)
    result = models.review_input(review)
    # result = NLP.review_input(review)
    lbl_output.configure(text=str(result))
    # if result >= 0.5:
    #     lbl_output.configure(text=str(result))
    # else:
    #     lbl_output.configure(text=str(result))

def down(e):
    if e.keycode == 13:
        clicked()
     


window.title("Welcome to Restaurant review rate")
lbl_input = Label(window, text="Customer Review")
txt_input = Text(window)
txt_input.grid(column=1, row = 0)
lbl_input.grid(column=0, row = 0) 
window.bind('<KeyPress>', down)

lbl_output = Label(window, text="Classification", bg="orange", fg="red")
txt_output = Text(window)
lbl_output.grid(column=1, row = 2)


btn = Button(window, text= "Classify", command=clicked)
btn.grid(column=1, row=1)
window.resizable()
window.mainloop()