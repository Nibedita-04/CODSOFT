import tkinter
import tkinter.messagebox
import random

#create root window
root = tkinter.Tk()

#change the root window background color
root.configure(bg="white")

#change the title
root.title("To-Do")

#change the window size
root.geometry("350x300")

#create an empty list
tasks=[]

#for testing purposes use a default list
# tasks = ["Call mom", "Buy a guitar", "Eat sushi"]

#create functions

def update_listbox():
    #Clear the current list
    clear_listbox()
    #Populate the listbox
    for task in tasks:
        lb_tasks.insert("end",task)
        
def clear_listbox():
    lb_tasks.delete(0,"end")

def add_task():
    #Get the task to add
    task = txt_input.get()
    #Make sure the task is not empty
    if task != "":
        #Append to the list
        tasks.append(task)
        #Update the listbox
        update_listbox()
    else:
        tkinter.messagebox.showwarning("Warning", "You need to enter the task")
    txt_input.delete(0,"end")

def delete_all():
    confirmed = tkinter.messagebox.askyesno("Please Confirm", "Do you really want to delete all ?")
    if confirmed == True:
        #Since we are changing the list, it needs to be global
        global tasks
        #Clear the tasks list
        tasks = []
        update_listbox()
    else:
        pass

    
def delete_one():
    #Get the  text of the currentlyy selected item
    task = lb_tasks.get("active")
    #Confirm it is in the list
    if task in tasks:
        tasks.remove(task)
    #Update the listbox
    update_listbox()

def sort_asc():
    #Sort the list
    tasks.sort()
    #Update the listbox
    update_listbox()

def sort_desc():
    #Sort the list
    tasks.sort()
    #Reverse the list
    tasks.reverse()
    #Update the listbox
    update_listbox()

def choose_random():
    #Choose a random task
    task = random.choice(tasks)
    #Update the display label
    lbl_display["text"] = task

def show_number_of_tasks():
    #get the number of tasks
    number_of_tasks = len(tasks)
    #Create and format the message
    msg = "Number of tasks : %s" %number_of_tasks
    #Update the display label
    lbl_display["text"] = msg


lbl_title = tkinter.Label(root, text="To-Do-List", bg="white")
lbl_title.grid(row=0,column=0, padx=5, pady=2)

lbl_display = tkinter.Label(root, text="", bg="white")
lbl_display.grid(row=0,column=1, padx=5, pady=2)

txt_input = tkinter.Entry(root, width=15, border="2px solid black")
txt_input.grid(row=1,column=1, padx=5, pady=2)

btn_add_task = tkinter.Button(root, text="Add Task", fg="black", bg="aqua", command=add_task, width=20)
btn_add_task.grid(row=1,column=0, padx=5, pady=2)

btn_del_all = tkinter.Button(root, text="Delete All", fg="black", bg="yellow", command=delete_all, width=20)
btn_del_all.grid(row=2,column=0, padx=5, pady=2)

btn_del_one = tkinter.Button(root, text="Delete", fg="black", bg="aqua", command=delete_one, width=20)
btn_del_one.grid(row=3,column=0, padx=5, pady=2)

btn_sort_asc = tkinter.Button(root, text="Sort (ASC)", fg="black", bg="yellow", command=sort_asc, width=20)
btn_sort_asc.grid(row=4,column=0, padx=5, pady=2)

btn_sort_desc = tkinter.Button(root, text="Sort (DESC)", fg="black", bg="aqua", command=sort_desc, width=20)
btn_sort_desc.grid(row=5,column=0, padx=5, pady=2)

btn_choose_random = tkinter.Button(root, text="Choose Random", fg="black", bg="yellow", command=choose_random, width=20)
btn_choose_random.grid(row=6,column=0, padx=5, pady=2)

btn_number_of_tasks = tkinter.Button(root, text="Number of Tasks", fg="black", bg="aqua", command=show_number_of_tasks, width=20)
btn_number_of_tasks.grid(row=7,column=0, padx=5, pady=2)

btn_exit = tkinter.Button(root, text="Exit", fg="white", bg="red", command=quit, width=20)
btn_exit.grid(row=8,column=0, padx=5, pady=2)

lb_tasks  = tkinter.Listbox(root, width=25, bg="#f4f4f4", border="2px solid black")
lb_tasks.grid(row=2,column=1,rowspan=7, padx=10, pady=10)

#start the main events loop
root.mainloop()