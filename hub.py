import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os

# --- Launch Functions ---

def launch_script(script_name, app_display_name):
    """Generic function to launch a Python script."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, script_name)
    python_executable = sys.executable

    try:
        # Use Popen to run the script as a separate, non-blocking process
        subprocess.Popen([python_executable, script_path])
        print(f"Launched: {python_executable} {script_path}")
    except FileNotFoundError:
        error_msg = (f"Error: '{script_name}' not found at expected location:\n{script_path}\n\n"
                     f"Please ensure '{script_name}' for the '{app_display_name}' "
                     "is in the same directory as this hub application.")
        print(error_msg)
        messagebox.showerror("Launch Error", error_msg)
    except Exception as e:
        error_msg = f"An unexpected error occurred while trying to launch '{script_name}' for '{app_display_name}':\n{e}"
        print(error_msg)
        messagebox.showerror("Launch Error", error_msg)

def launch_graphical_calculator():
    """Launches the g_calc.py script (Graphical Calculator)."""
    launch_script("g_calc.py", "Graphical Calculator")

def launch_ai_math_assistant():
    """Launches the ai_calc.py script (AI Math Assistant)."""
    launch_script("ai_calc.py", "AI Math Assistant")

def launch_unit_converter():
    """Launches the s_calc.py script (Unit Converter)."""
    launch_script("s_calc.py", "Unit Converter")

def launch_advanced_calculator():
    """Launches the a_calc.py script (Advanced Calculator)."""
    launch_script("a_calc.py", "Advanced Calculator")


# --- Create the main hub window ---
hub_window = tk.Tk()
hub_window.title("Application Hub")
hub_window.geometry("450x350")
hub_window.configure(bg="#e0e0e0")

# --- Styling ---
button_font = ("Arial", 12)
button_bg = "#f0f0f0"
button_active_bg = "#d9d9d9"
button_relief = tk.RAISED
button_padding_x = 10
button_padding_y = 10

# --- Main frame for buttons ---
button_frame = tk.Frame(hub_window, padx=20, pady=20, bg=hub_window.cget('bg'))
button_frame.pack(expand=True, fill=tk.BOTH)

# Configure grid layout for the frame
button_frame.grid_rowconfigure(0, weight=1)
button_frame.grid_rowconfigure(1, weight=1)
button_frame.grid_columnconfigure(0, weight=1)
button_frame.grid_columnconfigure(1, weight=1)

# --- Create the buttons ---

# Button 1: Graphical Calculator
btn_calculator = tk.Button(
    button_frame,
    text="Graphical Calculator",
    command=launch_graphical_calculator,
    font=button_font,
    bg=button_bg,
    activebackground=button_active_bg,
    relief=button_relief,
    padx=button_padding_x,
    pady=button_padding_y
)

# Button 2: AI Math Assistant
btn_ai_assistant = tk.Button(
    button_frame,
    text="AI Math Assistant", # Updated text
    command=launch_ai_math_assistant, # Updated command
    font=button_font,
    bg=button_bg,
    activebackground=button_active_bg,
    relief=button_relief,
    padx=button_padding_x,
    pady=button_padding_y
    # state=tk.DISABLED removed to enable the button
)

# Button 3: Unit Converter
btn_unit_converter = tk.Button(
    button_frame,
    text="Unit Converter", # Updated text
    command=launch_unit_converter, # Updated command
    font=button_font,
    bg=button_bg,
    activebackground=button_active_bg,
    relief=button_relief,
    padx=button_padding_x,
    pady=button_padding_y
    # state=tk.DISABLED removed
)

# Button 4: Advanced Calculator
btn_advanced_calc = tk.Button(
    button_frame,
    text="Advanced Calculator",
    command=launch_advanced_calculator,
    font=button_font,
    bg=button_bg,
    activebackground=button_active_bg,
    relief=button_relief,
    padx=button_padding_x,
    pady=button_padding_y
    # state=tk.DISABLED removed
)

# --- Arrange buttons in a 2x2 grid ---
# Using sticky="nsew" to make buttons expand to fill their grid cell
btn_calculator.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
btn_ai_assistant.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
btn_unit_converter.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
btn_advanced_calc.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

# Start the Tkinter event loop for the hub window
if __name__ == "__main__":
    hub_window.mainloop()
