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
        self.lbl_output.place(x = 560,y = 600)
        
        self.btn_classify = Button(window, text= "Classify", command = self.classify)
        self.btn_classify.place(x = 570,y = 550)
        
        self.my_menu = Menu(window)
        self.window.config(menu=self.my_menu)
        self.export_menu = Menu(self.my_menu)
        self.my_menu.add_cascade(label="Export", menu = self.export_menu)
        self.export_menu.add_command(label = "Top non stopword count", 
                                     command = self.export_top_non_stopword_count)
        self.export_menu.add_separator()
        self.export_menu.add_command(label = "Top stopword count", 
                                     command = self.export_top_stopword_count)
        self.export_menu.add_separator()
        self.export_menu.add_command(label = "Compare nonstopword and stopword at runtime", 
                                     command = self.export_total_stopword_appear_total_non_stopword_appear)
        
        self.export_menu.add_separator()
        self.export_menu.add_command(label = "Compare nonstopword and stopword", 
                                     command = self.export_stopword_and_non_stopword)

        
        

        self.select_model_menu = Menu(self.my_menu)
        self.my_menu.add_cascade(label="Select model", menu = self.select_model_menu)
        self.select_model_menu.add_command(label = "Artificial Neural Network", 
                                     command = self.ann_select)
        self.select_model_menu.add_separator()
        self.select_model_menu.add_command(label = "GaussianNB", 
                                     command = self.gauss_select)
        self.select_model_menu.add_separator()
        self.select_model_menu.add_command(label = "MultinomialNB", 
                                     command = self.multi_select)
        
        self.select_model_menu.add_separator()
        self.select_model_menu.add_command(label = "Support vector machines", 
                                     command = self.svc_select)

        

        
    def ann_select(self):
        self.window.title('ANN')
        self.model = ANLP.Model(model_path = r'model',is_ann = True)
    
    def gauss_select(self):
        self.window.title('GaussianNB')
        self.model = ANLP.Model(model_path = r'classifier\gaussianNB_classifier.pickle',is_ann = False)
    
    def multi_select(self):
        self.window.title('MultinomialNB')
        self.model = ANLP.Model(model_path = r'classifier\multinomialNB_classifier.pickle',is_ann = False)

    def svc_select(self):
        self.model = ANLP.Model(model_path = r'classifier\svc_classifier.pickle',is_ann = False)
        self.window.title('SVC')

    def popup_number(self):
        a=simpledialog.askinteger(title="Top", prompt="Enter number of words",parent=self.window )
        return a
        
    def export_top_stopword_count(self):
        a = self.popup_number()
        self.model.top_stopword_count(a)
        
    def export_top_non_stopword_count(self):
        a = self.popup_number()
        self.model.top_non_stopword_count(a)
    
    def export_total_stopword_appear_total_non_stopword_appear(self):
        self.model.total_stopword_appear_total_non_stopword_appear()
    
    def export_stopword_and_non_stopword(self):
        self.model.stopword_and_non_stopword()
        
    def command(self):
        pass
    
    def add_to_database(self):
        review = self.txt_input.get("1.0","end")
        review = review.replace("\n"," ")
        if len(review)==1:
            messagebox.showinfo(title="Status", message="Can't find the review in textbox")
        else:
            result = messagebox.askyesno(title='Confirm', 
                                          message='Does the Result correct?')
            
            if result == True:
                file = open("Restaurant_Reviews.tsv","a+")
                if (self.model.review_input(review)>=0.5):
                    review = review + "\t"+ "1\n"
                else:
                    review = review + "\t"+ "0\n"
            elif result == False:
                file = open("Restaurant_Reviews.tsv","a+")
                if (self.model.review_input(review)>=0.5):
                    review = review + "\t"+ "0\n"
                else:
                    review = review + "\t"+ "1\n"
            if(file.write(review)>0):
                messagebox.showinfo(title="Status", message='Success')
            else:
                messagebox.showinfo(title="Status", message='Fail')
            file.close()
            
    def classify(self):
        review = self.txt_input.get("1.0","end")
        result = self.model.review_input(review)
        # result = NLP.review_input(review)
        self.lbl_output.configure(text=str(result))
        # if result >= 0.5:
        #     lbl_output.configure(text=str(result))
        # else:
        #     lbl_output.configure(text=str(result))
        messagebox.showinfo(title="Result", message=str(result))
        self.add_to_database()
    
    def down(e):
        if e.keycode == 13:
            self.clicked()

def main():
    
    models = ANLP.Model(model_path=r'classifier\multinomialNB_classifier.pickle', is_ann = False)
    # models = ANLP.Model(model_path=r'model', is_ann = True)

    window=Tk()

    # window.bind('<KeyPress>', down)
    mywin=MyWindow(window,models)
    window.title('Hello Python')
    window.geometry("1120x650")
    window.mainloop()

if __name__ == '__main__':
    main()

