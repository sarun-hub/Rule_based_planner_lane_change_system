import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class CSVPlotterApp:
    def __init__(self, root):
        self.root = root
        self.root.title('ACC comparison Plotter')

        self.left_frame = tk.LabelFrame(root, text='Left Panel',padx=10, pady=10)
        self.right_frame = tk.LabelFrame(root, text='Right Panel',padx=10, pady=10)
        self.left_frame.grid(row = 0, column=0, padx=10, pady=10)
        self.right_frame.grid(row = 0, column=1, padx=10, pady=10)
        
        self.add_controls()

        self.left_canvas = None
        self.right_canvas = None

    def add_controls(self):
        tk.Button(self.left_frame, text="Select CSV", command=self.load_left_csv).pack()
        tk.Button(self.right_frame, text="Select CSV", command=self.load_right_csv).pack()

    def load_left_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            df = pd.read_csv(file_path)
            self.plot_csv(df, self.left_frame, file_path.split("/")[-1], panel='left')

    def load_right_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            df = pd.read_csv(file_path)
            self.plot_csv(df, self.right_frame, file_path.split("/")[-1], panel='right')

    def plot_csv(self, df, frame, title, panel='left'):
        if df.shape[1] < 2:
            tk.messagebox.showerror("Error", "CSV must have at least two columns.")
            return
        
        fig, axes = plt.subplots(2,1, figsize=(6,5), dpi=100)
        fig.suptitle(title)
        # fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.subplots_adjust(hspace=0.4, top=0.85)


        axes[0].plot(df.iloc[:,0],label=df.columns[0],color='blue')
        axes[0].set_title(df.columns[0])
        axes[0].set_xlabel('step')
        axes[0].set_ylabel('acceleration (m/s^2)')

        axes[1].plot(df.iloc[:,1],label=df.columns[1],color='green')
        axes[1].set_title(df.columns[1])
        axes[1].set_xlabel('step')
        axes[1].set_ylabel('acceleration (m/s^2)')

        for ax in axes:
            ax.grid(True)
            ax.set_ylim([-3,3])

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        plt.close(fig)

        if panel == 'left' and self.left_canvas:
            self.left_canvas.get_tk_widget().destroy()
        elif panel == 'right' and self.right_canvas:
            self.right_canvas.get_tk_widget().destroy()

        canvas.get_tk_widget().pack()
        if panel == 'left':
            self.left_canvas = canvas
        else:
            self.right_canvas = canvas

if __name__ == '__main__':
    root = tk.Tk()
    app = CSVPlotterApp(root)
    root.mainloop()

        