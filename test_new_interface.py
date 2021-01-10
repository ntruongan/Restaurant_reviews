from tkinter import *
from tkinter import simpledialog
import model as ANLP




class MyWindow:
    def __init__(self, window, models):
        self.model = models
        self.window = window
        self.lbl_input = Label(window, text="Customer Review")
        self.txt_input = Text(window)
        self.lbl_input.place(x = 0,y = 100)
        self.txt_input.place(x = 150,y = 000)
        
        self.lbl_output = Label(window, text="Classification", bg="orange", fg="red")
        self.txt_output = Text(window)
        self.lbl_output.place(x = 150,y = 000)
        
        self.my_menu = Menu(window)
        self.window.config(menu=self.my_menu)
        self.export_menu = Menu(self.my_menu)
        self.my_menu.add_cascade(label="Export", menu = self.export_menu)
        self.export_menu.add_command(label = "Top non stopword count", 
                                     command = self.export_top_non_stopword_count)
        self.export_menu.add_separator()
        self.export_menu.add_command(label = "Top stopword count", 
                                     command = self.export_top_stopword_count)

        
        
        self.btn_classify = Button(window, text= "Classify", command = self.command)
        self.btn_classify.place(x = 150,y = 000)
        
        # self.btn_chart_1 = Button(window, text= "Chart", command=self.popup_number)
        # self.btn_chart_1.place(x = 150,y = 200)
        # self.lbl1=Label(win, text='First number')
        # self.lbl2=Label(win, text='Second number')
        # self.lbl3=Label(win, text='Result')
        # self.t1=Entry(bd=3)
        # self.t2=Entry()
        # self.t3=Entry()
        # self.btn1 = Button(win, text='Add')
        # self.btn2=Button(win, text='Subtract')
        # self.lbl1.place(x=100, y=50)
        # self.t1.place(x=200, y=50)
        # self.lbl2.place(x=100, y=100)
        # self.t2.place(x=200, y=100)
        # self.b1=Button(win, text='Add', command=self.add)
        # self.b2=Button(win, text='Subtract')
        # self.b2.bind('<Button-1>', self.sub)
        # self.b1.place(x=100, y=150)
        # self.b2.place(x=200, y=150)
        # self.lbl3.place(x=100, y=200)
        # self.t3.place(x=200, y=200)
    
    def popup_number(self):
        a=simpledialog.askinteger(title="Top", prompt="Enter number of words",parent=self.window )
        
        return a
        
    def export_top_stopword_count(self):
        a = self.popup_number()
        self.model.top_stopword_count(a)
        
    def export_top_non_stopword_count(self):
        a = self.popup_number()
        self.model.top_non_stopword_count(a)
    
    def command(self):
        pass
    # def add_to_database():
    #     review = txt_input.get("1.0","end")
    #     review = review.replace("\n"," ")
    #     if len(review)==1:
    #         messagebox.showinfo(title="Status", message="Can't find the review in textbox")
    #     else:
    #         result = messagebox.askyesno(title='Confirm', 
    #                                      message='Does the Result correct?')
            
    #         if result == True:
    #             file = open("Restaurant_Reviews.tsv","a+")
    #             if (self.model.review_input(review)>=0.5):
    #                 review = review + "\t"+ "1\n"
    #             else:
    #                 review = review + "\t"+ "0\n"
    #         elif result == False:
    #             file = open("Restaurant_Reviews.tsv","a+")
    #             if (self.model.review_input(review)>=0.5):
    #                 review = review + "\t"+ "0\n"
    #             else:
    #                 review = review + "\t"+ "1\n"
    #         if(file.write(review)>0):
    #             messagebox.showinfo(title="Status", message='Success')
    #         else:
    #             messagebox.showinfo(title="Status", message='Fail')
    #         file.close()
            
    # def clicked():
    #     review = txt_input.get("1.0","end")
    #     result = self.model.review_input(review)
    #     # result = NLP.review_input(review)
    #     lbl_output.configure(text=str(result))
    #     # if result >= 0.5:
    #     #     lbl_output.configure(text=str(result))
    #     # else:
    #     #     lbl_output.configure(text=str(result))
    #     messagebox.showinfo(title="Result", message=str(result))
    #     self.add_to_database()
    
    # def down(e):
    #     if e.keycode == 13:
    #         self.clicked()

def main():
    
    models = ANLP.Model()
    window=Tk()

    # window.bind('<KeyPress>', down)
    mywin=MyWindow(window,models)
    window.title('Hello Python')
    window.geometry("1280x1000+10+10")
    window.mainloop()

if __name__ == '__main__':
    main()

