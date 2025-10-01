import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import re # Import regular expressions module
import scipy.optimize # For fsolve
import sympy  # For symbolic differentiation
from pathlib import Path

# --- Preprocessing Function for Implicit Multiplication ---
def preprocess_for_implicit_multiplication(expr_str, known_identifiers_list):
    """
    Preprocesses a mathematical string to insert implicit multiplication '*' signs.
    """
    if not isinstance(expr_str, str):
        return expr_str

    processed_expr = expr_str.replace(" ", "").replace("\t", "")
    sorted_identifiers = sorted(known_identifiers_list, key=len, reverse=True)
    id_pattern = "|".join(map(re.escape, sorted_identifiers))

    if not id_pattern:
        return processed_expr

    processed_expr = re.sub(r"(\d+)(" + id_pattern + r"|\()", r"\1*\2", processed_expr)
    processed_expr = re.sub(r"(\))(" + id_pattern + r"|\d|\()", r"\1*\2", processed_expr)
    non_func_ids_for_paren = [s for s in ['x', 'pi', 'e'] if s in known_identifiers_list]
    if non_func_ids_for_paren:
        non_func_id_pattern_for_paren = "|".join(map(re.escape, non_func_ids_for_paren))
        processed_expr = re.sub(r"(" + non_func_id_pattern_for_paren + r")(\()", r"\1*\2", processed_expr)
    for _ in range(2):
        processed_expr = re.sub(r"(" + id_pattern + r")(" + id_pattern + r")", r"\1*\2", processed_expr)
    if 'x' in known_identifiers_list:
         processed_expr = re.sub(r"(x)(\d)", r"\1*\2", processed_expr)
    return processed_expr

# --- Safe Evaluation Function (for numerical evaluation) ---
def safe_eval(func_str, x_val, additional_vars=None):
    """
    Safely evaluates a function string for numerical computation.
    """
    current_allowed_names = {
        "x": x_val,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan, "arctan2": np.arctan2,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        "exp": np.exp, "log": np.log, "log10": np.log10, "log2": np.log2,
        "sqrt": np.sqrt, "abs": np.abs, "power": np.power,
        "pi": np.pi, "e": np.e,
        "sign": np.sign, "ceil": np.ceil, "floor": np.floor, "sinc": np.sinc,
        "degrees": np.degrees, "radians": np.radians,
        "maximum": np.maximum, "minimum": np.minimum,
        "where": np.where,
        "logical_and": np.logical_and,
        "logical_or": np.logical_or,
        "logical_not": np.logical_not,
    }
    if additional_vars:
        current_allowed_names.update(additional_vars)
    known_identifiers_for_syntax_preprocessing = list(current_allowed_names.keys())
    processed_func_str = func_str
    if isinstance(func_str, str) and func_str.strip():
        processed_func_str = preprocess_for_implicit_multiplication(func_str, known_identifiers_for_syntax_preprocessing)
    return eval(processed_func_str, {"__builtins__": {
        "abs": abs, "round": round, "min": min, "max": max,
    }}, current_allowed_names)

# --- Main Application Class ---
class GraphicCalculatorApp:
    def __init__(self, master):
        self.master = master
        master.title("Graphical Calculator")
        master.geometry("1000x800")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        style.configure("TLabel", padding=5)
        style.configure("TEntry", padding=5)

        # Primary function data
        self.current_function_str = None
        self.current_x_values_for_plot = None
        self.current_y_values_for_plot = None

        self.plotted_artists = [] # Stores all artists (lines, scatters, fills, etc.)
        self.clicked_points_coords = [] # Stores (x,y) of points clicked on the graph

        # For managing second function and its intersection points
        self.second_function_line_artist = None
        self.intersection_point_artists = []


        self._generic_identifiers_for_preprocess = [
            "x", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2",
            "sinh", "cosh", "tanh", "exp", "log", "log10", "log2", "sqrt",
            "abs", "power", "pi", "e", "sign", "ceil", "floor", "sinc",
            "degrees", "radians", "maximum", "minimum", "where",
            "logical_and", "logical_or", "logical_not",
            "Abs", "Pow", "E", "ceiling", "asin", "acos", "atan"
        ]

        control_frame = ttk.Frame(master, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.plot_frame = ttk.Frame(master, padding="10")
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Input Section (Primary Function) ---
        input_section = ttk.LabelFrame(control_frame, text="Input & Draw Primary f(x)", padding="10")
        input_section.pack(fill=tk.X, pady=5)
        ttk.Label(input_section, text="f(x) or Points (x1,y1; x2,y2; ...):").pack(anchor='w')
        self.func_entry = ttk.Entry(input_section, width=35)
        self.func_entry.pack(fill=tk.X, pady=5)
        self.func_entry.insert(0, "sin(x)")
        draw_button = ttk.Button(input_section, text="Draw Primary", command=self.draw_input)
        draw_button.pack(fill=tk.X, pady=5)
        load_csv_button = ttk.Button(input_section, text="Load CSV Points", command=self.load_csv_points)
        load_csv_button.pack(fill=tk.X, pady=5)

        # --- Plot Range Section ---
        range_section = ttk.LabelFrame(control_frame, text="Plot Range", padding="10")
        range_section.pack(fill=tk.X, pady=5)
        ttk.Label(range_section, text="X Min:").grid(row=0, column=0, sticky='w')
        self.x_min_entry = ttk.Entry(range_section, width=10)
        self.x_min_entry.grid(row=0, column=1, padx=5)
        self.x_min_entry.insert(0, "-10")
        ttk.Label(range_section, text="X Max:").grid(row=1, column=0, sticky='w')
        self.x_max_entry = ttk.Entry(range_section, width=10)
        self.x_max_entry.grid(row=1, column=1, padx=5)
        self.x_max_entry.insert(0, "10")
        ttk.Label(range_section, text="Points:").grid(row=2, column=0, sticky='w')
        self.num_points_entry = ttk.Entry(range_section, width=10)
        self.num_points_entry.grid(row=2, column=1, padx=5)
        self.num_points_entry.insert(0, "500")

        # --- Analysis Section ---
        analysis_section = ttk.LabelFrame(control_frame, text="Analysis (on Primary f(x) or Points)", padding="10")
        analysis_section.pack(fill=tk.X, pady=5)

        deriv_button = ttk.Button(analysis_section, text="Draw Derivative f'(x)", command=self.draw_derivative)
        deriv_button.pack(fill=tk.X, pady=5)

        lin_reg_button = ttk.Button(analysis_section, text="Plot Linear Regression (Points)", command=self.plot_linear_regression)
        lin_reg_button.pack(fill=tk.X, pady=5)

        # Intersection with a second function
        ttk.Label(analysis_section, text="Second f(x) for Intersection:").pack(anchor='w', pady=(5,0))
        self.second_func_entry = ttk.Entry(analysis_section, width=35)
        self.second_func_entry.pack(fill=tk.X, pady=2)
        self.second_func_entry.insert(0, "0.5*x") # Example second function
        draw_intersect_button = ttk.Button(analysis_section, text="Draw 2nd & Find Intersections", command=self.draw_second_function_and_find_intersections)
        draw_intersect_button.pack(fill=tk.X, pady=5)


        ttk.Label(analysis_section, text="Integrate from x1:").pack(anchor='w', pady=(5,0))
        self.integrate_x1_entry = ttk.Entry(analysis_section, width=10)
        self.integrate_x1_entry.pack(fill=tk.X, pady=2)
        self.integrate_x1_entry.insert(0, "0")
        ttk.Label(analysis_section, text="to x2:").pack(anchor='w')
        self.integrate_x2_entry = ttk.Entry(analysis_section, width=10)
        self.integrate_x2_entry.pack(fill=tk.X, pady=2)
        self.integrate_x2_entry.insert(0, "pi")
        integrate_button = ttk.Button(analysis_section, text="Calculate Integral", command=self.calculate_integral)
        integrate_button.pack(fill=tk.X, pady=5)

        ttk.Label(analysis_section, text="Evaluate f(x) at x:").pack(anchor='w', pady=(5,0))
        self.eval_x_entry = ttk.Entry(analysis_section, width=10)
        self.eval_x_entry.pack(fill=tk.X, pady=2)
        self.eval_x_entry.insert(0, "1")
        eval_button = ttk.Button(analysis_section, text="Find y-value for x", command=self.evaluate_y_for_x)
        eval_button.pack(fill=tk.X, pady=5)

        ttk.Label(analysis_section, text="Find x for y-value:").pack(anchor='w', pady=(5,0))
        self.find_y_target_entry = ttk.Entry(analysis_section, width=10)
        self.find_y_target_entry.pack(fill=tk.X, pady=2)
        self.find_y_target_entry.insert(0, "0.5")
        find_x_button = ttk.Button(analysis_section, text="Find x-value(s) for y", command=self.find_x_for_y)
        find_x_button.pack(fill=tk.X, pady=5)

        # --- Plot Utilities ---
        util_section = ttk.LabelFrame(control_frame, text="Plot Utilities", padding="10")
        util_section.pack(fill=tk.X, pady=5)
        clear_button = ttk.Button(util_section, text="Clear Plot", command=self.clear_plot_full)
        clear_button.pack(fill=tk.X, pady=5)

        # --- Matplotlib Setup ---
        self.fig, self.ax = plt.subplots(facecolor='#f0f0f0')
        self.ax.set_facecolor('#ffffff')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.axhline(0, color='black', linewidth=1.2, zorder=1)
        self.ax.axvline(0, color='black', linewidth=1.2, zorder=1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.draw()

        # Connect Matplotlib events
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_click)


    def _on_mouse_motion(self, event):
        """Handles mouse motion over the plot to display coordinates."""
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                self.toolbar.set_message(f"x={x:.3f}, y={y:.3f}")
            else:
                self.toolbar.set_message("")
        else:
            self.toolbar.set_message("")

    def _on_mouse_click(self, event):
        """Handles mouse clicks on the plot to draw points."""
        if event.inaxes == self.ax and event.button == 1: # Left click
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                # Draw a point (black circle)
                clicked_point_artist, = self.ax.plot(x, y, 'ko', markersize=5, zorder=12, label="Clicked Point") # zorder to be on top
                self.plotted_artists.append(clicked_point_artist)
                self.clicked_points_coords.append((x, y)) # Store coordinates
                # Update legend if it exists, or create one
                handles, labels = self.ax.get_legend_handles_labels()
                # Avoid duplicate "Clicked Point" labels
                unique_labels = {}
                new_handles = []
                new_labels = []
                for handle, label in zip(handles, labels):
                    if label not in unique_labels:
                        unique_labels[label] = handle
                        new_handles.append(handle)
                        new_labels.append(label)
                if new_handles:
                    self.ax.legend(new_handles, new_labels, loc='upper left')
                self.canvas.draw_idle() # Use draw_idle for better performance with frequent updates

    def _get_plot_params(self):
        try:
            x_min_str = self.x_min_entry.get()
            x_max_str = self.x_max_entry.get()
            x_min = safe_eval(x_min_str, None)
            x_max = safe_eval(x_max_str, None)
            num_points = int(self.num_points_entry.get())
            if num_points <= 1:
                messagebox.showerror("Error", "Number of points must be greater than 1.")
                return None
            if float(x_min) >= float(x_max):
                messagebox.showerror("Error", "X Min must be less than X Max.")
                return None
            return float(x_min), float(x_max), int(num_points)
        except Exception as e:
            messagebox.showerror("Plot Range Error", f"Invalid input in plot range fields: {e}\nEnsure you use valid numbers or constants like 'pi', 'e', or expressions like '2*pi'.")
            return None

    def _clear_all_plotted_artists(self):
        """Removes all artists stored in self.plotted_artists from the axes and clears clicked points."""
        for artist in self.plotted_artists:
            try:
                artist.remove()
            except (AttributeError, ValueError):
                pass
        self.plotted_artists.clear()
        self.clicked_points_coords.clear() # Clear stored clicked point coordinates

        if self.second_function_line_artist:
            try: self.second_function_line_artist.remove()
            except (AttributeError, ValueError): pass
            self.second_function_line_artist = None
        for artist in self.intersection_point_artists:
            try: artist.remove()
            except (AttributeError, ValueError): pass
        self.intersection_point_artists.clear()

        if self.ax.legend_:
            try: self.ax.legend_.remove()
            except (AttributeError, ValueError): pass
            self.ax.legend_ = None


    def clear_plot_full(self):
        self._clear_all_plotted_artists()
        self.current_function_str = None
        self.current_x_values_for_plot = None
        self.current_y_values_for_plot = None
        # self.clicked_points_coords.clear() # Already handled by _clear_all_plotted_artists
        self.second_function_line_artist = None
        self.intersection_point_artists = []
        self.ax.relim()
        self.ax.autoscale_view(True,True,True)
        self.ax.set_title("")
        self.canvas.draw()
        messagebox.showinfo("Info", "Plot cleared.")

    def parse_points(self, points_str):
        x_coords = []
        y_coords = []
        pairs = points_str.split(';')
        for pair_str in pairs:
            pair_str = pair_str.strip()
            if not pair_str: continue
            try:
                x_str, y_str = pair_str.split(',')
                x_coords.append(safe_eval(x_str.strip(), None))
                y_coords.append(safe_eval(y_str.strip(), None))
            except ValueError:
                raise ValueError(f"Invalid point format: '{pair_str}'. Expected 'x,y'.")
            except Exception as e:
                raise ValueError(f"Error evaluating point coordinate '{pair_str}': {e}")
        return np.array(x_coords, dtype=float), np.array(y_coords, dtype=float)



    def _plot_points_dataset(self, x_data, y_data, x_range_hint, label, color='green'):
        if x_data.size == 0 or y_data.size == 0:
            messagebox.showerror("Input Error", "No valid points found.")
            return False
        scatter = self.ax.scatter(x_data, y_data, label=label, color=color, zorder=5)
        self.plotted_artists.append(scatter)
        self.ax.set_title("Scatter Plot of Points")

        if x_range_hint:
            x_min_for_plot, x_max_for_plot = x_range_hint
        else:
            x_min_for_plot = float(np.min(x_data))
            x_max_for_plot = float(np.max(x_data))

        view_x_min = min(x_min_for_plot, np.min(x_data) - abs(np.min(x_data) * 0.1 + 1))
        view_x_max = max(x_max_for_plot, np.max(x_data) + abs(np.max(x_data) * 0.1 + 1))

        prior_artists = len(self.plotted_artists) > 1 or len(self.ax.lines) > 0
        current_y_lim = self.ax.get_ylim()
        base_y_min = current_y_lim[0] if prior_artists else float(np.min(y_data))
        base_y_max = current_y_lim[1] if prior_artists else float(np.max(y_data))
        view_y_min = min(base_y_min, np.min(y_data) - abs(np.min(y_data) * 0.1 + 1))
        view_y_max = max(base_y_max, np.max(y_data) + abs(np.max(y_data) * 0.1 + 1))

        self.ax.set_xlim(view_x_min, view_x_max)
        self.ax.set_ylim(view_y_min, view_y_max)

        self.current_function_str = None
        self.current_x_values_for_plot = x_data
        self.current_y_values_for_plot = y_data
        return True

    def _refresh_legend(self):
        if self.plotted_artists:
            handles, labels = self.ax.get_legend_handles_labels()
            if handles:
                self.ax.legend(handles, labels, loc='upper left')

    
    def draw_input(self):
        self._clear_all_plotted_artists() # This will also clear self.clicked_points_coords
        input_str = self.func_entry.get().strip()
        if not input_str:
            messagebox.showwarning("Input Error", "Input field for primary function is empty.")
            return

        plot_params = self._get_plot_params()
        if not plot_params:
            return

        x_min_for_plot, x_max_for_plot, num_points = plot_params

        try:
            is_points_input = ';' in input_str or                               (',' in input_str and input_str.count(';') == 0 and                                not any(func_name in input_str for func_name in ["sin", "cos", "tan", "log", "exp", "sqrt", "where", "power", "abs"]))

            if is_points_input:
                try:
                    x_data, y_data = self.parse_points(input_str)
                    preview = input_str if len(input_str) <= 30 else f"{input_str[:30]}..."
                    label = f"Input Points: {preview}"
                    if not self._plot_points_dataset(x_data, y_data, (x_min_for_plot, x_max_for_plot), label, color='green'):
                        return
                except ValueError as e:
                    messagebox.showerror("Input Error", f"Could not parse as points: {e}\nIf this was meant to be a function, please check syntax.")
                    return
            else:
                self._draw_function_str(input_str, x_min_for_plot, x_max_for_plot, num_points, color='blue', is_primary=True)

            self._refresh_legend()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Drawing Error", f"An error occurred: {e}")
            self.current_function_str = None
            self.current_x_values_for_plot = None
            self.current_y_values_for_plot = None


    def load_csv_points(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV with x,y columns",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            return

        x_values = []
        y_values = []
        header_skipped = False
        try:
            with open(file_path, newline='') as csv_file:
                reader = csv.reader(csv_file)
                for row_index, row in enumerate(reader, start=1):
                    if not row:
                        continue
                    cells = [cell.strip() for cell in row]
                    if not cells[0] or cells[0].startswith('#'):
                        continue
                    if len(cells) < 2:
                        continue
                    try:
                        x_val = float(cells[0])
                        y_val = float(cells[1])
                    except ValueError:
                        if not header_skipped:
                            header_skipped = True
                            continue
                        raise ValueError(f"Row {row_index} is not numeric: {row}")
                    x_values.append(x_val)
                    y_values.append(y_val)
        except Exception as exc:
            messagebox.showerror("CSV Load Error", f"Could not read '{file_path}': {exc}")
            return

        if not x_values:
            messagebox.showerror("CSV Load Error", "No numeric data rows found in the selected file.")
            return

        plot_params = self._get_plot_params()
        if not plot_params:
            return
        x_min_for_plot, x_max_for_plot, _ = plot_params

        self._clear_all_plotted_artists()

        x_data = np.array(x_values, dtype=float)
        y_data = np.array(y_values, dtype=float)
        label = f"CSV Points: {Path(file_path).name}"
        if not self._plot_points_dataset(x_data, y_data, (x_min_for_plot, x_max_for_plot), label, color='purple'):
            return

        self.func_entry.delete(0, tk.END)
        self.func_entry.insert(0, f"csv:{Path(file_path).name}")

        self._refresh_legend()
        self.canvas.draw()
        messagebox.showinfo("CSV Loaded", f"Loaded {len(x_values)} points from '{Path(file_path).name}'.")
    def _draw_function_str(self, func_str, x_min, x_max, num_points, color='blue', label_prefix="f(x) = ", is_primary=False):
        x_values = np.linspace(float(x_min), float(x_max), int(num_points))
        with np.errstate(divide='ignore', invalid='ignore'):
            y_values = safe_eval(func_str, x_values)
        if np.isscalar(y_values):
            y_values = np.full_like(x_values, y_values)
        y_values[~np.isfinite(y_values)] = np.nan
        line_label = f"{label_prefix}{func_str}"
        line, = self.ax.plot(x_values, y_values, label=line_label, color=color, zorder=10 if is_primary else 6)
        self.plotted_artists.append(line)
        if is_primary:
            self.current_function_str = func_str
            self.current_x_values_for_plot = x_values
            self.current_y_values_for_plot = y_values
            self.ax.set_title(f"f(x) = {func_str}")
        self.ax.set_xlim(x_min, x_max)
        finite_y = y_values[np.isfinite(y_values)]
        if finite_y.size > 0:
            y_data_min, y_data_max = np.min(finite_y), np.max(finite_y)
            y_range = y_data_max - y_data_min
            if y_range < 1e-9:
                y_padding = 1.0
            else:
                y_padding = 0.1 * y_range
            current_y_lims = self.ax.get_ylim()
            new_y_min = min(y_data_min - y_padding, current_y_lims[0] if not is_primary and len(self.ax.lines) > 1 else y_data_min - y_padding) # Check > 1 as primary line is already there
            new_y_max = max(y_data_max + y_padding, current_y_lims[1] if not is_primary and len(self.ax.lines) > 1 else y_data_max + y_padding)
            self.ax.set_ylim(new_y_min, new_y_max)
        elif is_primary :
            self.ax.autoscale(enable=True, axis='y', tight=False)
        return line

    def draw_derivative(self):
        if self.current_x_values_for_plot is None or self.current_y_values_for_plot is None:
            messagebox.showwarning("Derivative Error", "No data plotted. Please draw something first.")
            return
        if len(self.current_x_values_for_plot) < 2:
            messagebox.showwarning("Derivative Error", "Not enough points to calculate derivative (need at least 2).")
            return
        if self.current_function_str:
            try:
                x_sym = sympy.Symbol('x')
                processed_for_sympy = preprocess_for_implicit_multiplication(self.current_function_str, self._generic_identifiers_for_preprocess)
                sympy_ready_expr_str = processed_for_sympy.replace('power(', 'Pow(')
                sympy_ready_expr_str = re.sub(r'log10\((.*?)\)', r'log(\1, 10)', sympy_ready_expr_str)
                sympy_ready_expr_str = re.sub(r'log2\((.*?)\)', r'log(\1, 2)', sympy_ready_expr_str)
                sympy_ready_expr_str = sympy_ready_expr_str.replace('abs(', 'Abs(')
                sympy_ready_expr_str = sympy_ready_expr_str.replace('ceil(', 'ceiling(')
                sympy_namespace = {
                    'x': x_sym, 'pi': sympy.pi, 'E': sympy.E, 'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan,
                    'asin': sympy.asin, 'acos': sympy.acos, 'atan': sympy.atan, 'atan2': sympy.atan2,
                    'sinh': sympy.sinh, 'cosh': sympy.cosh, 'tanh': sympy.tanh, 'exp': sympy.exp, 'log': sympy.log,
                    'sqrt': sympy.sqrt, 'Abs': sympy.Abs, 'Pow': sympy.Pow, 'sign': sympy.sign,
                    'ceiling': sympy.ceiling, 'floor': sympy.floor, 'sinc': sympy.sinc
                }
                expr = sympy.sympify(sympy_ready_expr_str, locals=sympy_namespace)
                derivative_expr = sympy.diff(expr, x_sym)
                derivative_str_legend = str(derivative_expr)
                lambdify_modules = ['numpy', {'Abs': np.abs, 'ceiling': np.ceil, 'sinc': np.sinc}]
                derivative_func_numeric = sympy.lambdify(x_sym, derivative_expr, modules=lambdify_modules)
                y_derivative_values = derivative_func_numeric(self.current_x_values_for_plot)
                if np.isscalar(y_derivative_values):
                    y_derivative_values = np.full_like(self.current_x_values_for_plot, y_derivative_values)
                y_derivative_values[~np.isfinite(y_derivative_values)] = np.nan
                plot_label = f"d/dx ({self.current_function_str[:20]}...) = {derivative_str_legend[:30]}"
                if len(self.current_function_str) > 20 or len(derivative_str_legend) > 30 : plot_label += "..."
                line, = self.ax.plot(self.current_x_values_for_plot, y_derivative_values, label=plot_label, color='purple', linestyle='-.', zorder=8)
                self.plotted_artists.append(line)
                self.ax.legend(loc='upper left')
                self.canvas.draw()
                return
            except Exception as sym_e:
                messagebox.showwarning("Symbolic Differentiation Failed", f"Could not compute symbolic derivative for '{self.current_function_str}': {sym_e}\nFalling back to numerical differentiation.")
        try:
            current_x_for_num_deriv = self.current_x_values_for_plot
            current_y_for_num_deriv = self.current_y_values_for_plot
            label_prefix_numeric = f"d/dx ({self.current_function_str[:20]}...)" if self.current_function_str else "Derivative of points"
            label_numeric = f"{label_prefix_numeric} (numerical)"
            if self.current_function_str is None:
                if not np.all(np.diff(current_x_for_num_deriv) > 0):
                    if current_x_for_num_deriv.ndim == 1 and current_y_for_num_deriv.ndim == 1:
                        sort_indices = np.argsort(current_x_for_num_deriv)
                        current_x_for_num_deriv = current_x_for_num_deriv[sort_indices]
                        current_y_for_num_deriv = current_y_for_num_deriv[sort_indices]
                        if len(np.unique(current_x_for_num_deriv)) < len(current_x_for_num_deriv):
                             messagebox.showwarning("Derivative Error", "Scatter points have duplicate x-values after sorting. Cannot compute numerical derivative.")
                             return
                    else:
                        messagebox.showwarning("Derivative Error", "Point data is not 1D, cannot sort for numerical derivative.")
                        return
            y_derivative_numerical = np.gradient(current_y_for_num_deriv, current_x_for_num_deriv)
            y_derivative_numerical[~np.isfinite(y_derivative_numerical)] = np.nan
            line, = self.ax.plot(current_x_for_num_deriv, y_derivative_numerical, label=label_numeric, color='red', linestyle='--', zorder=9)
            self.plotted_artists.append(line)
            self.ax.legend(loc='upper left')
            self.canvas.draw()
        except Exception as num_e:
            messagebox.showerror("Numerical Derivative Error", f"Could not calculate/draw numerical derivative: {num_e}")

    def plot_linear_regression(self):
        all_x_points = []
        all_y_points = []

        # Collect points from primary input if they are actual points
        if self.current_function_str is None and \
           self.current_x_values_for_plot is not None and \
           self.current_y_values_for_plot is not None:
            all_x_points.extend(list(self.current_x_values_for_plot))
            all_y_points.extend(list(self.current_y_values_for_plot))

        # Collect points from clicks
        if self.clicked_points_coords:
            clicked_x, clicked_y = zip(*self.clicked_points_coords)
            all_x_points.extend(list(clicked_x))
            all_y_points.extend(list(clicked_y))

        if len(all_x_points) < 2:
            messagebox.showwarning("Linear Regression Error", "Not enough point data (from input or clicks) to perform linear regression. Need at least 2 points.")
            return

        try:
            x_points_np = np.array(all_x_points)
            y_points_np = np.array(all_y_points)

            slope, intercept = np.polyfit(x_points_np, y_points_np, 1)

            # Determine regression line range based on the actual points used
            if x_points_np.size > 0:
                x_reg_min = np.min(x_points_np)
                x_reg_max = np.max(x_points_np)
            else: # Should not happen due to len(all_x_points) < 2 check, but as fallback
                plot_params = self._get_plot_params()
                if plot_params:
                    x_reg_min, x_reg_max, _ = plot_params
                else:
                    x_reg_min, x_reg_max = self.ax.get_xlim()


            # If all points are coincident on x, polyfit might still give a slope (vertical line)
            # which is fine, but x_reg_line needs at least two distinct points if x_reg_min == x_reg_max
            if x_reg_min == x_reg_max:
                x_reg_line = np.array([x_reg_min - 0.5, x_reg_max + 0.5]) # Create a small range if all x are same
            else:
                x_reg_line = np.array([x_reg_min, x_reg_max])
            
            y_reg_line = slope * x_reg_line + intercept

            reg_label = f"Lin. Reg: y={slope:.2f}x + {intercept:.2f}"
            line, = self.ax.plot(x_reg_line, y_reg_line, label=reg_label, color='orange', linestyle=':', linewidth=2, zorder=7)
            self.plotted_artists.append(line)
            
            handles, labels = self.ax.get_legend_handles_labels()
            self.ax.legend(handles, labels, loc='upper left')
            
            self.canvas.draw()
            messagebox.showinfo("Linear Regression", f"Linear regression line plotted for available points.\nEquation: y = {slope:.4f}x + {intercept:.4f}")
        except Exception as e:
            messagebox.showerror("Linear Regression Error", f"Could not perform or plot linear regression: {e}")


    def draw_second_function_and_find_intersections(self):
        if self.current_function_str is None or self.current_x_values_for_plot is None:
            messagebox.showwarning("Intersection Error", "Please draw a primary function first.")
            return
        second_func_str = self.second_func_entry.get().strip()
        if not second_func_str:
            messagebox.showwarning("Input Error", "Second function field is empty.")
            return
        plot_params = self._get_plot_params()
        if not plot_params: return
        x_min_plot, x_max_plot, num_points = plot_params
        if self.second_function_line_artist:
            try:
                self.second_function_line_artist.remove()
                self.plotted_artists.remove(self.second_function_line_artist) # Also remove from general list
            except (ValueError, AttributeError): pass
            self.second_function_line_artist = None
        for artist in self.intersection_point_artists:
            try:
                artist.remove()
                self.plotted_artists.remove(artist) # Also remove from general list
            except (ValueError, AttributeError): pass
        self.intersection_point_artists = []
        try:
            x_for_second_func = self.current_x_values_for_plot # Use x-values of primary func for denser check
            with np.errstate(divide='ignore', invalid='ignore'):
                y_values_second_func = safe_eval(second_func_str, x_for_second_func)
            if np.isscalar(y_values_second_func):
                y_values_second_func = np.full_like(x_for_second_func, y_values_second_func)
            y_values_second_func[~np.isfinite(y_values_second_func)] = np.nan
            self.second_function_line_artist, = self.ax.plot(x_for_second_func, y_values_second_func,
                                                            label=f"g(x) = {second_func_str}",
                                                            color='darkcyan', linestyle='--', zorder=7)
            self.plotted_artists.append(self.second_function_line_artist) # Add to general list
            def diff_func(x_candidate):
                try:
                    y1 = safe_eval(self.current_function_str, x_candidate)
                    y2 = safe_eval(second_func_str, x_candidate)
                    return y1 - y2
                except Exception: return np.inf
            found_intersections_x = set()
            num_guesses_intersect = max(50, int(num_points / 5)) # More guesses for fsolve
            x_search_range = np.linspace(x_min_plot, x_max_plot, num_guesses_intersect)
            for x_guess in x_search_range:
                try:
                    x_solution, _, ier, _ = scipy.optimize.fsolve(diff_func, x_guess, full_output=True, xtol=1.49012e-08)
                    if ier == 1: # Solution found
                        y1_at_sol = safe_eval(self.current_function_str, x_solution[0])
                        y2_at_sol = safe_eval(second_func_str, x_solution[0])
                        if np.isfinite(y1_at_sol) and np.isfinite(y2_at_sol) and abs(y1_at_sol - y2_at_sol) < 1e-5: # Check they are close
                            if x_min_plot <= x_solution[0] <= x_max_plot: # Check within plot bounds
                                found_intersections_x.add(round(x_solution[0], 6)) # Add rounded to avoid near duplicates
                except Exception: continue
            if not found_intersections_x:
                messagebox.showinfo("Intersections", f"No intersections found between '{self.current_function_str}' and '{second_func_str}' in the range [{x_min_plot:.2f}, {x_max_plot:.2f}].")
            else:
                intersect_msg = f"Found intersection(s) at x ≈\n"
                sorted_intersections_x = sorted(list(found_intersections_x))
                for x_intersect in sorted_intersections_x:
                    y_intersect = safe_eval(self.current_function_str, x_intersect) # Or average of both funcs
                    intersect_msg += f"  x = {x_intersect:.4f}, y = {y_intersect:.4f}\n"
                    point, = self.ax.plot(x_intersect, y_intersect, 'rX', markersize=10, label=f"Intersect ({x_intersect:.2f})", zorder=11)
                    self.intersection_point_artists.append(point)
                    self.plotted_artists.append(point) # Add to general list
                messagebox.showinfo("Intersections Found", intersect_msg)
            self.ax.legend(loc='upper left')
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Intersection Error", f"Could not find intersections: {e}")

    def calculate_integral(self):
        if self.current_function_str is None:
            messagebox.showwarning("Integral Error", "No function plotted. Please draw a function first to integrate.")
            return
        try:
            x1_str = self.integrate_x1_entry.get()
            x2_str = self.integrate_x2_entry.get()
            x1 = float(safe_eval(x1_str, None))
            x2 = float(safe_eval(x2_str, None))
            if x1 >= x2:
                messagebox.showerror("Integral Error", "x1 must be less than x2 for integration.")
                return
            num_integral_points = max(2000, int(self.num_points_entry.get()) * 4) # High resolution for trapz
            integrate_x = np.linspace(x1, x2, num_integral_points)
            with np.errstate(divide='ignore', invalid='ignore'):
                integrate_y = safe_eval(self.current_function_str, integrate_x)
            if np.isscalar(integrate_y): # Handle constant functions
                integrate_y = np.full_like(integrate_x, integrate_y)
            if not np.all(np.isfinite(integrate_y)):
                messagebox.showwarning("Integral Warning", "Function is undefined or results in non-finite values (NaN/Inf) within the integration range. Result may be inaccurate or NaN.")
            integral_value = np.trapz(integrate_y[np.isfinite(integrate_y)], integrate_x[np.isfinite(integrate_y)]) # Use only finite parts for trapz
            messagebox.showinfo("Integral Result", f"Integral of {self.current_function_str} from {x1_str} to {x2_str} is approx: {integral_value:.6f}")
            fill_artist = self.ax.fill_between(integrate_x, integrate_y, 0, color='skyblue', alpha=0.4, label=f"Integral Area [{x1_str}, {x2_str}]")
            self.plotted_artists.append(fill_artist) # Add to general list
            self.ax.legend(loc='upper left')
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Integral Error", f"Could not calculate integral: {e}")

    def evaluate_y_for_x(self):
        if self.current_function_str is None:
            messagebox.showwarning("Evaluation Error", "No function plotted. Please draw a function first.")
            return
        try:
            x_val_str = self.eval_x_entry.get()
            x_val = float(safe_eval(x_val_str, None))
            y_val = safe_eval(self.current_function_str, x_val)
            messagebox.showinfo("Function Evaluation", f"For f(x) = {self.current_function_str}:\nAt x = {x_val_str} ({x_val:.4f}), y = {y_val:.6f}")
            point_artist_tuple = self.ax.plot(x_val, y_val, 'mo', markersize=8, label=f"Eval at x={x_val_str}")
            self.plotted_artists.extend(point_artist_tuple) # Add to general list
            self.ax.legend(loc='upper left')
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Evaluation Error", f"Could not evaluate function: {e}")

    def find_x_for_y(self):
        if self.current_function_str is None:
            messagebox.showwarning("Find X Error", "No function plotted. Please draw a function first.")
            return
        try:
            y_target_str = self.find_y_target_entry.get()
            y_target = float(safe_eval(y_target_str, None))
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid y-value target: {e}")
            return
        def func_to_solve(x_candidate):
            try:
                val = safe_eval(self.current_function_str, x_candidate)
                return val - y_target
            except Exception: return np.inf
        plot_params = self._get_plot_params()
        if not plot_params:
            if self.current_x_values_for_plot is not None and len(self.current_x_values_for_plot) > 0:
                 x_min_plot, x_max_plot = np.min(self.current_x_values_for_plot), np.max(self.current_x_values_for_plot)
            else:
                x_min_plot, x_max_plot = -10, 10 # Default if no plot params and no current x values
                messagebox.showwarning("Find X Warning", "Using default x-range (-10 to 10) for search as plot range is not set.")
        else:
            x_min_plot, x_max_plot, _ = plot_params
        found_x_values = set()
        num_guesses = 20 # Number of initial guesses for fsolve
        initial_guesses = np.linspace(x_min_plot, x_max_plot, num_guesses)
        for x_guess in initial_guesses:
            try:
                x_solution, infodict, ier, mesg = scipy.optimize.fsolve(func_to_solve, x_guess, full_output=True, xtol=1.49012e-08)
                if ier == 1: # Solution found
                    y_at_solution = safe_eval(self.current_function_str, x_solution[0])
                    if abs(y_at_solution - y_target) < 1e-5: # Verify solution
                        found_x_values.add(round(x_solution[0], 6)) # Add rounded value
            except Exception: continue
        if not found_x_values:
            messagebox.showinfo("Find X Result", f"No x-values found for y = {y_target_str} ({y_target:.4f}) within the current x-range and search attempts.")
            return
        visible_solutions = sorted([val for val in found_x_values if x_min_plot <= val <= x_max_plot])
        if not visible_solutions:
            messagebox.showinfo("Find X Result", f"No x-values found for y = {y_target_str} ({y_target:.4f}) within the visible plot x-range ({x_min_plot:.2f} to {x_max_plot:.2f}). Some solutions might be outside this range.")
            return
        result_message = f"Found x-value(s) for y = {y_target_str} ({y_target:.4f}):\n"
        result_message += "\n".join([f"x ≈ {val:.6f}" for val in visible_solutions])
        messagebox.showinfo("Find X Result", result_message)
        for x_sol in visible_solutions:
            point_artist_tuple = self.ax.plot(x_sol, y_target, 'bD', markersize=7, label=f"y={y_target_str} at x≈{x_sol:.2f}")
            self.plotted_artists.extend(point_artist_tuple) # Add to general list
        line_artist = self.ax.axhline(y_target, color='purple', linestyle=':', linewidth=1, label=f"Target y={y_target_str}")
        self.plotted_artists.append(line_artist) # Add to general list
        self.ax.legend(loc='upper left')
        self.canvas.draw()

# --- Main execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = GraphicCalculatorApp(root)
    root.mainloop()
