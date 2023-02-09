from tkinter import Tk, StringVar, IntVar, N, W, E, S, Scrollbar
from tkinter import ttk
from joblib import load, dump
import pandas as pd


class ReadPolusaArticles:
    
    def __init__(self, root):
        self.df = load('2018_2')
        print(self.df.head())

        root.title = "Polusa Dataset Articles"

        mainframe = ttk.Frame(root, padding="12 12 12 12")
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Data decleration
        self.article = IntVar()
        self.article_header = StringVar()
        self.article_text = StringVar()
        self.article_leaning = StringVar()
        self.rating = StringVar()
        self.saved_hint = StringVar()
        self.count = IntVar()
        article_entry = ttk.Entry(mainframe, width=7, textvariable=self.article)
        article_entry.grid(column=2, row=1, sticky=(W, E))
        

        # row 1 (Article)
        ttk.Label(mainframe, text="Article Row").grid(column=1, row=1, sticky=W)
        ttk.Button(mainframe, text="Get", command=self.readArticle).grid(column=3, row=1, sticky=W)

        # row 2 (Header)
        ttk.Label(mainframe, text="Heading").grid(column=2, row=2, sticky=W)
        ttk.Label(mainframe, textvariable=self.article_header, width= 140).grid(column=3, row=2, sticky=W)

        # row 3 (Article Body)
        ttk.Label(mainframe, text="Body").grid(column=2, row=3, sticky=W)
        ttk.Label(mainframe, textvariable=self.article_text, width=140, wraplength=1200, font=("Arial", 16)).grid(column=3, row=3, sticky=W)

        # row 4 (Article Leaning)
        # ttk.Label(mainframe, text="Political Leaning").grid(column=2, row=4, sticky=W)
        # ttk.Label(mainframe, textvariable=self.article_leaning).grid(column=3, row=4, sticky=W)

        # row 5 (Own Leaning)
        ttk.Label(mainframe, text="My Rating").grid(column=1, row=5, sticky=W)
        rating = ttk.Entry(mainframe, width=7, textvariable=self.rating)
        rating.grid(column=2, row=5, sticky=(W, E))
        ttk.Button(mainframe, text="Rate", command=self.saveRating).grid(column=3, row=5, stick=W)
        ttk.Label(mainframe, textvariable=self.saved_hint).grid(column=3, row=5, sticky=W)

        # row 6 (Navigation)
        ttk.Button(mainframe, text="Previous Article", command=self.prevArticle).grid(column=1, row=6, sticky=W)
        ttk.Button(mainframe, text="Next Article", command=self.nextArticle).grid(column=2, row=6, sticky=W)

        # row 8 (Statistics)
        ttk.Label(mainframe, text="Rated").grid(column=1, row=8, sticky=W)
        ttk.Label(mainframe, textvariable=self.count).grid(column=2, row=8, sticky=W)

        for child in mainframe.winfo_children(): 
            child.grid_configure(padx=5, pady=5)

        article_entry.focus()
        root.bind("<Return>", self.readArticle)


    def readArticle(self, *args):

        # print(self.df.head())
        self.saved_hint.set('')
        read_row = self.df.iloc[self.article.get()]

        self.article_header.set(read_row.headline)
        self.article_text.set(read_row.body)
        self.article_leaning.set(read_row.political_leaning)
        self.rating.set(read_row.rating)
        self.stats()

    
    def saveRating(self, *args):
        self.df.at[self.article.get(), 'rating'] = self.rating.get()
        dump(self.df, '2018_2', compress=4)
        self.saved_hint.set('Saved Rating!')
        self.stats()


    def nextArticle(self, *args):
        self.article.set(self.article.get()+1)
        self.readArticle()

    
    def prevArticle(self, *args):
        self.article.set(self.article.get()-1)
        self.readArticle()

    def stats(self, *args):
        filter = self.df["rating"] != "UNDEFINED"
        count = self.df[self.df["rating"]!="UNDEFINED"].count()
        self.count.set(count["rating"])


root = Tk()
ReadPolusaArticles(root)
root.mainloop()