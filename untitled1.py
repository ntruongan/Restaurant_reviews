# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 21:59:26 2021

@author: ntruo
"""

import tkinter as tk

LARGE_FONT= ("Verdana", 12)
HEIGHT = 768
WIDTH = 1366


class MainApp():
    def __init__(self, master):
        self.master = master
        self.master.title("Sales System") 
        self.master.geometry("%dx%d+0+0" % (WIDTH, HEIGHT)) 

        self.frames = {}

        start_page = StartPage(master)

        self.frames[StartPage] = start_page

        start_page.grid(row=0, column=0, sticky="nsew")
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()


class base_frame(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        tk.Frame.__init__(master, *args, **kwargs)

        b_image = tk.PhotoImage(file='background.png')
        b_label = tk.Label(self, image=b_image)
        b_label.image = b_image
        b_label.place(x=0, y=0, relwidth=1, relheight=1)

        topleft_label = tk.Label(self, bg='black', fg='white', text="Welcome - Login Screen", justify='left', anchor="w", font="Verdana 12")
        topleft_label.place(relwidth=0.5, relheight=0.05, relx=0.25, rely=0, anchor='n')

class StartPage(base_frame):

    def __init__(self, parent):
        super().__init__(self, parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

def main():
    root = tk.Tk() # MainApp()
    main_app = MainApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()