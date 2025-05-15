import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import re # Import regular expressions module
import scipy.optimize # For fsolve
import sympy # For symbolic differentiation
# scipy.stats is not strictly needed if using np.polyfit, but good to have for other stats
# from scipy import stats 

# --- Preprocessing Function for Implicit Multiplication ---
def preprocess_for_implicit_multiplication(expr_str, known_identifiers_list):
    """
    Preprocesses a mathematical string to insert implicit multiplication '*' signs.
    """
    if not isinstance(expr_str, str): 
        return expr_str
        
    processed_expr = expr_str.replace(" ", "").replace("\t", "")
    # Sort by length to match longer identifiers first (e.g., "arcsin" before "sin")
    sorted_identifiers = sorted(known_identifiers_list, key=len, reverse=True)
    id_pattern = "|".join(map(re.escape, sorted_identifiers))

    if not id_pattern: 
        return processed_expr

    # Digit followed by identifier or '(' -> insert '*'
    processed_expr = re.sub(r"(\d+)(" + id_pattern + r"|\()", r"\1*\2", processed_expr)
    # ')' followed by identifier, digit or '(' -> insert '*'
    processed_expr = re.sub(r"(\))(" + id_pattern + r"|\d|\()", r"\1*\2", processed_expr)

    # Identifier (like x, pi, e) followed by '(' -> insert '*'
    non_func_ids_for_paren = [s for s in ['x', 'pi', 'e'] if s in known_identifiers_list]
    if non_func_ids_for_paren:
        non_func_id_pattern_for_paren = "|".join(map(re.escape, non_func_ids_for_paren))
        processed_expr = re.sub(r"(" + non_func_id_pattern_for_paren + r")(\()", r"\1*\2", processed_expr)

    # Identifier followed by identifier -> insert '*' (applied twice for chains)
    for _ in range(2): 
        processed_expr = re.sub(r"(" + id_pattern + r")(" + id_pattern + r")", r"\1*\2", processed_expr)
    
    # 'x' followed by digit -> insert '*'
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

    # Get identifiers for preprocessing (syntax correction)
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

        self.current_function_str = None
        self.current_x_values_for_plot = None 
        self.current_y_values_for_plot = None 
        self.plotted_artists = [] 

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

        # --- Input Section ---
        input_section = ttk.LabelFrame(control_frame, text="Input & Draw", padding="10")
        input_section.pack(fill=tk.X, pady=5)
        ttk.Label(input_section, text="f(x) or Points (x1,y1; x2,y2; ...):").pack(anchor='w')
        self.func_entry = ttk.Entry(input_section, width=35) 
        self.func_entry.pack(fill=tk.X, pady=5)
        self.func_entry.insert(0, "sin(x)") 
        draw_button = ttk.Button(input_section, text="Draw", command=self.draw_input)
        draw_button.pack(fill=tk.X, pady=5)

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
        analysis_section = ttk.LabelFrame(control_frame, text="Analysis", padding="10")
        analysis_section.pack(fill=tk.X, pady=5)

        deriv_button = ttk.Button(analysis_section, text="Draw Derivative f'(x)", command=self.draw_derivative)
        deriv_button.pack(fill=tk.X, pady=5)
        
        lin_reg_button = ttk.Button(analysis_section, text="Plot Linear Regression (Points)", command=self.plot_linear_regression)
        lin_reg_button.pack(fill=tk.X, pady=5)

        ttk.Label(analysis_section, text="Integrate from x1:").pack(anchor='w')
        self.integrate_x1_entry = ttk.Entry(analysis_section, width=10)
        self.integrate_x1_entry.pack(fill=tk.X, pady=2)
        self.integrate_x1_entry.insert(0, "0")
        ttk.Label(analysis_section, text="to x2:").pack(anchor='w')
        self.integrate_x2_entry = ttk.Entry(analysis_section, width=10)
        self.integrate_x2_entry.pack(fill=tk.X, pady=2)
        self.integrate_x2_entry.insert(0, "pi") 
        integrate_button = ttk.Button(analysis_section, text="Calculate Integral", command=self.calculate_integral)
        integrate_button.pack(fill=tk.X, pady=5)

        ttk.Label(analysis_section, text="Evaluate f(x) at x:").pack(anchor='w')
        self.eval_x_entry = ttk.Entry(analysis_section, width=10)
        self.eval_x_entry.pack(fill=tk.X, pady=2)
        self.eval_x_entry.insert(0, "1")
        eval_button = ttk.Button(analysis_section, text="Find y-value for x", command=self.evaluate_y_for_x)
        eval_button.pack(fill=tk.X, pady=5)

        ttk.Label(analysis_section, text="Find x for y-value:").pack(anchor='w')
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

    def _clear_previous_plots(self):
        for artist in self.plotted_artists:
            artist.remove()
        self.plotted_artists.clear()
        if self.ax.legend_: 
            self.ax.legend_.remove() 
            self.ax.legend_ = None

    def clear_plot_full(self):
        self._clear_previous_plots()
        self.current_function_str = None
        self.current_x_values_for_plot = None
        self.current_y_values_for_plot = None
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

    def draw_input(self):
        self._clear_previous_plots() 
        input_str = self.func_entry.get().strip()
        if not input_str:
            messagebox.showwarning("Input Error", "Input field is empty.")
            return

        plot_params = self._get_plot_params()
        if not plot_params: return 

        x_min_for_plot, x_max_for_plot, num_points = plot_params
        
        try:
            is_points_input = ';' in input_str or \
                              (',' in input_str and input_str.count(';') == 0 and \
                               not any(func_name in input_str for func_name in ["sin", "cos", "tan", "log", "exp", "sqrt", "where", "power", "abs"]))
            
            if is_points_input:
                try:
                    x_data, y_data = self.parse_points(input_str)
                    if x_data.size == 0:
                        messagebox.showerror("Input Error", "No valid points found.")
                        return

                    line = self.ax.scatter(x_data, y_data, label=f"Points: {input_str[:30]}...", color='green', zorder=5)
                    self.plotted_artists.append(line)
                    self.ax.set_title(f"Scatter Plot of Points")
                    
                    view_x_min = min(x_min_for_plot, np.min(x_data) - abs(np.min(x_data)*0.1 + 1)) 
                    view_x_max = max(x_max_for_plot, np.max(x_data) + abs(np.max(x_data)*0.1 + 1)) 
                    
                    current_y_lim = self.ax.get_ylim()
                    view_y_min = min(current_y_lim[0] if len(self.ax.lines) > 0 else np.min(y_data), np.min(y_data) - abs(np.min(y_data)*0.1 + 1))
                    view_y_max = max(current_y_lim[1] if len(self.ax.lines) > 0 else np.max(y_data), np.max(y_data) + abs(np.max(y_data)*0.1 + 1))

                    self.ax.set_xlim(view_x_min, view_x_max)
                    self.ax.set_ylim(view_y_min, view_y_max)
                    
                    self.current_function_str = None 
                    self.current_x_values_for_plot = x_data 
                    self.current_y_values_for_plot = y_data

                except ValueError as e: 
                    messagebox.showerror("Input Error", f"Could not parse as points: {e}\nIf this was meant to be a function, please check syntax.")
                    return 
            else: 
                self._draw_function_str(input_str, x_min_for_plot, x_max_for_plot, num_points)

            if self.plotted_artists: 
                self.ax.legend(loc='upper left')
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Drawing Error", f"An error occurred: {e}")
            self.current_function_str = None
            self.current_x_values_for_plot = None
            self.current_y_values_for_plot = None

    def _draw_function_str(self, func_str, x_min, x_max, num_points):
        self.current_x_values_for_plot = np.linspace(float(x_min), float(x_max), int(num_points))
        
        with np.errstate(divide='ignore', invalid='ignore'): 
            self.current_y_values_for_plot = safe_eval(func_str, self.current_x_values_for_plot)
        
        if np.isscalar(self.current_y_values_for_plot):
            self.current_y_values_for_plot = np.full_like(self.current_x_values_for_plot, self.current_y_values_for_plot)

        self.current_y_values_for_plot[~np.isfinite(self.current_y_values_for_plot)] = np.nan

        line, = self.ax.plot(self.current_x_values_for_plot, self.current_y_values_for_plot, label=f"f(x) = {func_str}", color='blue', zorder=10)
        self.plotted_artists.append(line)
        self.current_function_str = func_str 
        self.ax.set_title(f"f(x) = {func_str}")
        
        self.ax.set_xlim(x_min, x_max)
        
        finite_y = self.current_y_values_for_plot[np.isfinite(self.current_y_values_for_plot)]
        if finite_y.size > 0:
            y_data_min, y_data_max = np.min(finite_y), np.max(finite_y)
            y_range = y_data_max - y_data_min
            if y_range < 1e-9: 
                y_padding = 1.0
            else:
                y_padding = 0.1 * y_range
            self.ax.set_ylim(y_data_min - y_padding, y_data_max + y_padding)
        else: 
            self.ax.autoscale(enable=True, axis='y', tight=False)

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
                    'x': x_sym, 'pi': sympy.pi, 'E': sympy.E,
                    'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan,
                    'asin': sympy.asin, 'acos': sympy.acos, 'atan': sympy.atan, 'atan2': sympy.atan2,
                    'sinh': sympy.sinh, 'cosh': sympy.cosh, 'tanh': sympy.tanh,
                    'exp': sympy.exp, 'log': sympy.log, 
                    'sqrt': sympy.sqrt, 'Abs': sympy.Abs, 'Pow': sympy.Pow,
                    'sign': sympy.sign, 'ceiling': sympy.ceiling, 'floor': sympy.floor,
                    'sinc': sympy.sinc 
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
        """Plots a linear regression line for the current point data."""
        if self.current_function_str is not None:
            messagebox.showwarning("Linear Regression Error", "Linear regression is for plotted points, not for a symbolic function. Please draw points first.")
            return
        if self.current_x_values_for_plot is None or self.current_y_values_for_plot is None or len(self.current_x_values_for_plot) < 2:
            messagebox.showwarning("Linear Regression Error", "Not enough point data to perform linear regression (need at least 2 points).")
            return

        try:
            x_points = self.current_x_values_for_plot
            y_points = self.current_y_values_for_plot

            # Perform linear regression: y = mx + c
            # np.polyfit returns [slope, intercept] for degree 1
            slope, intercept = np.polyfit(x_points, y_points, 1)
            
            # Generate y-values for the regression line
            # Extend the line slightly beyond the min/max of points for better visualization
            x_reg_min = np.min(x_points)
            x_reg_max = np.max(x_points)
            # x_reg_line = np.array([x_reg_min - 0.1*(x_reg_max-x_reg_min), x_reg_max + 0.1*(x_reg_max-x_reg_min)]) # Extend slightly
            x_reg_line = np.array([x_reg_min, x_reg_max]) # Or just use the range of points
            
            y_reg_line = slope * x_reg_line + intercept

            # Plot the regression line
            reg_label = f"Lin. Reg: y={slope:.2f}x + {intercept:.2f}"
            line, = self.ax.plot(x_reg_line, y_reg_line, label=reg_label, color='orange', linestyle=':', linewidth=2, zorder=7)
            self.plotted_artists.append(line)
            
            self.ax.legend(loc='upper left')
            self.canvas.draw()
            messagebox.showinfo("Linear Regression", f"Linear regression line plotted.\nEquation: y = {slope:.4f}x + {intercept:.4f}")

        except Exception as e:
            messagebox.showerror("Linear Regression Error", f"Could not perform or plot linear regression: {e}")


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

            num_integral_points = max(2000, int(self.num_points_entry.get()) * 4) 
            integrate_x = np.linspace(x1, x2, num_integral_points)
            with np.errstate(divide='ignore', invalid='ignore'):
                integrate_y = safe_eval(self.current_function_str, integrate_x)
            if np.isscalar(integrate_y): 
                integrate_y = np.full_like(integrate_x, integrate_y)
            if not np.all(np.isfinite(integrate_y)):
                messagebox.showwarning("Integral Warning", "Function is undefined or results in non-finite values (NaN/Inf) within the integration range. Result may be inaccurate or NaN.")
            integral_value = np.trapz(integrate_y, integrate_x)
            messagebox.showinfo("Integral Result", f"Integral of {self.current_function_str} from {x1_str} to {x2_str} is approx: {integral_value:.6f}")
            fill_artist = self.ax.fill_between(integrate_x, integrate_y, 0, color='skyblue', alpha=0.4, label=f"Integral Area [{x1_str}, {x2_str}]")
            self.plotted_artists.append(fill_artist) 
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
            self.plotted_artists.extend(point_artist_tuple) 
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
            except Exception:
                return np.inf 

        plot_params = self._get_plot_params()
        if not plot_params: 
            if self.current_x_values_for_plot is not None and len(self.current_x_values_for_plot) > 0:
                 x_min_plot, x_max_plot = np.min(self.current_x_values_for_plot), np.max(self.current_x_values_for_plot)
            else: 
                x_min_plot, x_max_plot = -10, 10 
                messagebox.showwarning("Find X Warning", "Using default x-range (-10 to 10) for search as plot range is not set.")
        else:
            x_min_plot, x_max_plot, _ = plot_params

        found_x_values = set() 
        num_guesses = 20 
        initial_guesses = np.linspace(x_min_plot, x_max_plot, num_guesses)

        for x_guess in initial_guesses:
            try:
                x_solution, infodict, ier, mesg = scipy.optimize.fsolve(func_to_solve, x_guess, full_output=True, xtol=1.49012e-08)
                if ier == 1: 
                    y_at_solution = safe_eval(self.current_function_str, x_solution[0])
                    if abs(y_at_solution - y_target) < 1e-5: 
                        found_x_values.add(round(x_solution[0], 6))
            except Exception:
                continue 

        if not found_x_values:
            messagebox.showinfo("Find X Result", f"No x-values found for y = {y_target_str} ({y_target:.4f}) within the current x-range and search attempts.")
            return

        visible_solutions = sorted([val for val in found_x_values if x_min_plot <= val <= x_max_plot])
        if not visible_solutions:
            messagebox.showinfo("Find X Result", f"No x-values found for y = {y_target_str} ({y_target:.4f}) within the visible plot x-range.")
            return

        result_message = f"Found x-value(s) for y = {y_target_str} ({y_target:.4f}):\n"
        result_message += "\n".join([f"x ≈ {val:.6f}" for val in visible_solutions])
        messagebox.showinfo("Find X Result", result_message)

        for x_sol in visible_solutions:
            point_artist_tuple = self.ax.plot(x_sol, y_target, 'bD', markersize=7, label=f"y={y_target_str} at x≈{x_sol:.2f}") 
            self.plotted_artists.extend(point_artist_tuple)
        
        line_artist = self.ax.axhline(y_target, color='purple', linestyle=':', linewidth=1, label=f"Target y={y_target_str}")
        self.plotted_artists.append(line_artist)
        self.ax.legend(loc='upper left')
        self.canvas.draw()

# --- Main execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = GraphicCalculatorApp(root)
    root.mainloop()
