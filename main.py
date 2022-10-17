import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import tree
from sklearn.metrics import accuracy_score
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from sklearn.metrics import accuracy_score

loan_dataset = pd.read_csv('loan_data.csv')

print(loan_dataset.head())

print(loan_dataset.shape)

print(loan_dataset.describe())

print(loan_dataset.isnull().sum())

loan_dataset = loan_dataset.dropna()

print(loan_dataset['Dependents'].value_counts())

loan_dataset['Dependents'] = loan_dataset['Dependents'].replace("3+", 4)

print(loan_dataset['Dependents'].value_counts())

categorical_columns = ['Gender', 'Married', 'Dependents',
                       'Education', 'Self_Employed', 'Property_Area',
                       'Credit_History','Loan_Amount_Term']

fig,axes = plt.subplots(4,2,figsize=(12,15))
for idx,cat_col in enumerate(categorical_columns):
    row,col = idx//2,idx%2
    sns.countplot(x=cat_col,data=loan_dataset,hue='Loan_Status',ax=axes[row,col])

plt.subplots_adjust(hspace=1)
plt.show()

loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)

loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

print(loan_dataset.head())

X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = loan_dataset['Loan_Status']
columns_names = list(X.columns)
print(columns_names)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X = pd.DataFrame(X, columns = columns_names)
print(X)
print(Y)

X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)


# logistic model

logisticmodel = LogisticRegression()

logisticmodel.fit(X_train, Y_train)

predict = logisticmodel.predict(X_test)

print('Accuracy on logistic test data : ', accuracy_score(predict, Y_test))

def logistic_model(test):
    test = np.array(test).reshape(1, 11)
    test = pd.DataFrame(test, columns=columns_names)
    test = scaler.transform(test)
    test = pd.DataFrame(test, columns=columns_names)
    result = logisticmodel.predict(test)
    print("result logistic_model ",result[0])

# SVM model

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, Y_train)

X_test_prediction = classifier.predict(X_test)

test_data_accuray = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on svm test data : ', test_data_accuray)

def SVM_model(test):
    test = np.array(test).reshape(1, 11)
    test = pd.DataFrame(test, columns=columns_names)
    test = scaler.transform(test)
    test = pd.DataFrame(test, columns=columns_names)
    result = classifier.predict(test)
    print("result of SVM model ", result[0])


# Tree Desion

tree_model = tree.DecisionTreeClassifier()

tree_model.fit(X_train, Y_train)

tree_predict = tree_model.predict(X_test)

print('Accuracy on  tree training data : ', accuracy_score(tree_predict, Y_test))

def Dicesion_Tree(test):
    test = np.array(test).reshape(1, 11)
    test = pd.DataFrame(test, columns=columns_names)
    test = scaler.transform(test)
    test = pd.DataFrame(test, columns=columns_names)
    result = tree_model.predict(test)
    print("result of tree_model ", result[0])


win = Tk()
win.title("loan predict window")
win.geometry("700x600")
win.resizable(False, False)

gender = Label(win, text="Gender", font=("Arial", 12)).place(x=50, y=50)
gender_text = StringVar(win)
gendercombo = ttk.Combobox(win, values=["male", "female"], textvariable=gender_text ,state="readonly").place(width=100, height=35, x=130, y=50)

Married = Label(win, text="Married", font=("Arial", 12)).place(x=250, y=50)
Married_text = StringVar(win)
Marriedcombo = ttk.Combobox(win, values=["Yes", "No"], textvariable=Married_text ,state="readonly").place(width=100, height=35, x=330, y=50)

Education = Label(win, text="Education", font=("Arial", 12)).place(x=450, y=50)
Education_text = StringVar(win)
Educationcombo = ttk.Combobox(win, values=["Graduated", "Not Graduated"], textvariable=Education_text ,state="readonly").place(width=120, height=35, x=550, y=50)


Self_Employed = Label(win, text="Self_Employed", font=("Arial", 12)).place(x=50, y=150)
Self_Employed_text = StringVar(win)
Self_Employedcombo = ttk.Combobox(win, values=["Yes", "No"], textvariable=Self_Employed_text  ,state="readonly").place(width=100, height=35, x=190, y=150)



ApplicantIncome = Label(win, text="ApplicantIncome", font=("Arial", 12)).place(x=320, y=150)
ApplicantIncome_Entry =Entry(win, font=("Arial", 16))
ApplicantIncome_Entry.place(width=130, height=35, x=480, y=150)


CoapplicantIncome = Label(win, text="CoapplicantIncome", font=("Arial", 12)).place(x=50, y=250)
CoapplicantIncome_Entry =Entry(win, font=("Arial", 16))
CoapplicantIncome_Entry.place(width=130, height=35, x=230, y=250)


LoanAmount = Label(win, text="LoanAmount", font=("Arial", 12)).place(x=390, y=250)
LoanAmount_Entry =Entry(win, font=("Arial", 16))
LoanAmount_Entry.place(width=130, height=35, x=520, y=250)

Loan_Amount_Term = Label(win, text="Loan_Amount_Term", font=("Arial", 12)).place(x=50, y=350)
Loan_Amount_Term_Entry =Entry(win, font=("Arial", 16))
Loan_Amount_Term_Entry.place(width=130, height=35, x=240, y=350)

Property_Area = Label(win, text="Property_Area", font=("Arial", 12)).place(x=390, y=350)
Property_Area_text = StringVar(win)
Property_Areacombo = ttk.Combobox(win, values=["Urban", "Semiurban","Rural"], textvariable=Property_Area_text ,state="readonly").place(width=120, height=35, x=540, y=350)


Credit_History = Label(win, text="Credit_History", font=("Arial", 12)).place(x=50, y=450)
Credit_History_text = StringVar(win)
Credit_History_Areacombo = ttk.Combobox(win, values=["1", "0"], textvariable=Credit_History_text ,state="readonly").place(width=100, height=35, x=210, y=450)


Dependents = Label(win, text="Dependents", font=("Arial", 12)).place(x=350, y=450)
Dependents_Entry = Entry(win, font=("Arial", 16))
Dependents_Entry.place(width=130, height=35, x=480, y=450)


def predict():
    if gender_text.get() != "" and Married_text.get() != "" and Education_text.get() != "" and ApplicantIncome_Entry.get() != "" and Self_Employed_text.get() != "" and CoapplicantIncome_Entry.get() != "" and LoanAmount_Entry.get() != "" and Loan_Amount_Term_Entry.get() != "" and Property_Area_text.get() != "" and Credit_History_text.get() != ""and Dependents_Entry.get()!="":
        try:
            int(ApplicantIncome_Entry.get())
            int(CoapplicantIncome_Entry.get())
            int(LoanAmount_Entry.get())
            int(Loan_Amount_Term_Entry.get())
            int(Dependents_Entry.get())
            test = []
            if gender_text.get() == "male":
                test.append(1)
            elif gender_text.get() == "female":
                test.append(0)

            if Married_text.get() == "Yes":
                test.append(1)
            elif Married_text.get() == "No":
                test.append(0)

            if int(Dependents_Entry.get()) >= 4:
                test.append(4)
            elif int(Dependents_Entry.get()) >= 0 and int(Dependents_Entry.get()) < 4:
                test.append(int(Dependents_Entry.get()))
            else:
                raise Exception()

            if Education_text.get() == "Graduated":
                test.append(1)
            elif Education_text.get() == "Not Graduated":
                test.append(0)

            if Self_Employed_text.get() == "Yes":
                test.append(1)
            elif Self_Employed_text.get() == "No":
                test.append(0)

            if int(ApplicantIncome_Entry.get()) >= 0:
                test.append(int(ApplicantIncome_Entry.get()))
            else:
                raise Exception()

            if int(CoapplicantIncome_Entry.get()) >= 0:
                test.append(int(CoapplicantIncome_Entry.get()))
            else:
                raise Exception()

            if int(LoanAmount_Entry.get()) >= 0:
                test.append(int(LoanAmount_Entry.get()))
            else:
                raise Exception()

            if int(Loan_Amount_Term_Entry.get()) >= 0:
                test.append(int(Loan_Amount_Term_Entry.get()))
            else:
                raise Exception()

            test.append(int(Credit_History_text.get()))

            if Property_Area_text.get() == "Urban":
                test.append(2)
            elif Property_Area_text.get() == "Rural":
                test.append(0)
            elif Property_Area_text.get() == "Semiurban":
                test.append(1)
            print(test)

            logistic_model(test)
            SVM_model(test)
            Dicesion_Tree(test)
        except:
            messagebox.showerror("fill boxes that has number with number not text", parent=win)
    else:
        messagebox.showerror("fill all boxs", parent=win)

predict_button = Button(win, command=predict ,text="predict")
predict_button.place(width=150, height=50, x=300, y=500)


win.mainloop()