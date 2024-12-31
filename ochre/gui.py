import sys
import tkinter as tk
from tkinter import filedialog

from ochre.cli import create_dwelling

# Script to create OCHRE executables


def gui_basic():
    root = tk.Tk()
    root.withdraw()

    input_path = filedialog.askdirectory(
        title="Select OCHRE Simulation Folder"
    )

    if not input_path:
        print("No simulation folder chosen, exiting.")
        return

    dwelling = create_dwelling(input_path)
    dwelling.simulate()


def set_entry_text(entry, text):
    entry.delete(0, tk.END)
    entry.insert(0, text)


def browse_file(entry):
    file_name = filedialog.askopenfilename()
    if file_name:
        set_entry_text(entry, file_name)


def browse_folder(entry):
    file_name = filedialog.askdirectory()
    if file_name:
        set_entry_text(entry, file_name)


class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, string):
        self.widget.configure(state="normal")
        self.widget.insert("end", string, (self.tag,))
        self.widget.configure(state="disabled")


def gui_detailed():
    root = tk.Tk()
    root.title("Enter inputs for OCHRE run")

    # add prompts for each input
    global row
    row = 0
    def make_input(label, default="", width=None, is_file=False, is_folder=False):
        global row
        label = tk.Label(root, text=label)
        label.grid(row=row, column=0, sticky="e")

        entry = tk.Entry(root, width=width)
        entry.grid(row=row, column=1, sticky="w")
        if default:
            set_entry_text(entry, default)

        if is_file:
            button = tk.Button(root, text="Browse", command=lambda: browse_file(entry))
            button.grid(row=row, column=2)
        if is_folder:
            button = tk.Button(root, text="Browse", command=lambda: browse_folder(entry))
            button.grid(row=row, column=2)

        row += 1
        return entry

    inputs = {
        "input_path": make_input("Input Path (required):", width=40, is_folder=True),
        "name": make_input("Simulation Name:", default="ochre"),
        "hpxml_file": make_input("HPXML File Name:", default="home.xml"),
        "hpxml_schedule_file": make_input("HPXML Schedule File Name:", default="in.schedules.csv"),
        "weather_file_or_path": make_input("Weather File or Path Name:", width=40, is_file=True),
        "output_path": make_input("Output Path:", width=40, is_folder=True),
        "verbosity": make_input("Verbosity (0-9):", default=3),
        "start_year": make_input("Start Year:", default=2018),
        "start_month": make_input("Start Month:", default=1),
        "start_day": make_input("Start Day:", default=1),
        "time_res": make_input("Time Resolution (minutes):", default=60),
        "duration": make_input("Duration (days):", default=365),
        "initialization_time": make_input("Initialization Duration (days):", default=1),
    }

    def get_inputs_and_run():
        # parse number and empty values
        input_values = {}
        for key, entry in inputs.items():
            val = entry.get()
            if not val:
                continue
            try:
                val = int(val)
            except ValueError:
                pass
            input_values[key] = val

        if "input_path" not in input_values:
            print("Must specify an input path.")
            return

        dwelling = create_dwelling(**input_values)
        dwelling.simulate()
        

    submit_button = tk.Button(root, text="Run OCHRE", command=get_inputs_and_run)
    submit_button.grid(row=row, column=0, columnspan=2)
    submit_button = tk.Button(root, text="Exit", command=root.destroy)
    submit_button.grid(row=row, column=2)
    row += 1

    output = tk.Text(root, wrap="word")
    output.grid(row=row, column=0, columnspan=3, sticky="nsew")
    sys.stdout = TextRedirector(output, "stdout")
    sys.stderr = TextRedirector(output, "stderr")

    root.rowconfigure(tuple(range(row)), weight=1)
    root.columnconfigure((0, 1, 2), weight=1)

    root.mainloop()

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        gui_basic()
    elif sys.argv[1] == "basic":
        gui_basic()
    elif sys.argv[1] == "detailed":
        gui_detailed()
    else:
        print("Unknown argument:", sys.argv[1])
