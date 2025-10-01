import tkinter as tk
from tkinter import messagebox
import sympy

# --- Global SymPy symbols and functions for parsing ---
x_sym = sympy.Symbol('x')
SYMPY_GLOBALS = {
    "pi": sympy.pi,
    "e": sympy.E,
    "sqrt": sympy.sqrt,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "log": sympy.log, 
    "ln": sympy.log,
    "log10": lambda val: sympy.log(val, 10),
    "exp": sympy.exp,
    "abs": sympy.Abs,
    "x": x_sym,
    "I": sympy.I,
    "i": sympy.I
}

class AdvancedCalculator:
    def __init__(self, master):
        self.master = master
        master.title("Advanced Calculator")
        master.geometry("400x550")
        master.resizable(False, False)

        self.expression = ""
        self.shift_active = False
        self.shift_button = None
        self.input_text = tk.StringVar()

        input_frame = tk.Frame(master, bd=0, relief=tk.RIDGE, bg="#333333")
        input_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.input_field = tk.Entry(input_frame, textvariable=self.input_text, font=('arial', 24, 'bold'),
                                    bd=10, insertwidth=2, width=14, borderwidth=4,
                                    justify='right', state='readonly', readonlybackground='white')
        self.input_field.pack(ipady=10)

        btns_frame = tk.Frame(master, bg="#cccccc")
        btns_frame.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        buttons = [
            ('(', 0, 0, 1, 1, 'input', '('), (')', 0, 1, 1, 1, 'input', ')'),
            ('pi', 0, 2, 1, 1, 'input', 'pi'), ('e', 0, 3, 1, 1, 'input', 'e'),

            ('AC', 1, 0, 1, 1, 'clear', None), ('DEL', 1, 1, 1, 1, 'del', None),
            ('^', 1, 2, 1, 1, 'input', '**'), ('/', 1, 3, 1, 1, 'input', '/'),

            ('7', 2, 0, 1, 1, 'input', '7'), ('8', 2, 1, 1, 1, 'input', '8'),
            ('9', 2, 2, 1, 1, 'input', '9'), ('*', 2, 3, 1, 1, 'input', '*'),

            ('4', 3, 0, 1, 1, 'input', '4'), ('5', 3, 1, 1, 1, 'input', '5'),
            ('6', 3, 2, 1, 1, 'input', '6'), ('-', 3, 3, 1, 1, 'input', '-'),

            ('1', 4, 0, 1, 1, 'input', '1'), ('2', 4, 1, 1, 1, 'input', '2'),
            ('3', 4, 2, 1, 1, 'input', '3'), ('+', 4, 3, 1, 1, 'input', '+'),

            ('0', 5, 0, 1, 1, 'input', '0'), ('.', 5, 1, 1, 1, 'input', '.'),
            ('x', 5, 2, 1, 1, 'input', 'x'), ('=', 5, 3, 1, 1, 'calculate', None),

            ('Shift', 6, 0, 1, 1, 'toggle_shift', None), ('sqrt()', 6, 1, 1, 1, 'special', 'sqrt('),
            ('Solve x', 6, 2, 2, 1, 'solve_x', None)
        ]
        for (text, r, c, cs, rs, ftype, val) in buttons:
            btn = tk.Button(btns_frame, text=text, font=('arial', 14),
                            fg="black", width=6, height=2, relief='raised', borderwidth=2,
                            command=lambda t=text, ft=ftype, v=val: self.button_action(t, ft, v))
            btn.grid(row=r, column=c, columnspan=cs, rowspan=rs, sticky="nsew", padx=2, pady=2)
            if ftype in ('calculate', 'solve_x'):
                btn.config(bg="#4CAF50", fg="white")
            elif ftype in ('clear', 'del'):
                btn.config(bg="#f44336", fg="white")
            elif ftype == 'toggle_shift':
                btn.config(bg="#607D8B", fg="white")
                self.shift_button = btn
            elif ftype == 'input' and val in ['+', '-', '*', '/', '**']:
                btn.config(bg="#FF9800")
            else:
                btn.config(bg="#e0e0e0")

        self._create_shift_panel(btns_frame)

        for i in range(7):
            btns_frame.grid_rowconfigure(i, weight=1)
        btns_frame.grid_rowconfigure(7, weight=0)
        for i in range(4):
            btns_frame.grid_columnconfigure(i, weight=1)

    def button_action(self, text, ftype, value):
        if ftype == 'input':
            self.expression += str(value)
        elif ftype == 'special':
            self.expression += str(value)
        elif ftype == 'clear':
            self.expression = ""
        elif ftype == 'del':
            self.expression = self.expression[:-1]
        elif ftype == 'calculate':
            self.evaluate_expression()
        elif ftype == 'solve_x':
            self.solve_for_x()
        elif ftype == 'toggle_shift':
            self.toggle_shift()
            return
        self.input_text.set(self.expression)

    def _create_shift_panel(self, parent):
        shift_options = [
            ('sin', 'sin('),
            ('cos', 'cos('),
            ('tan', 'tan('),
            ('i', 'I'),
            ('abs', 'abs('),
            ('log', 'log('),
            ('ln', 'ln('),
            ('pi', 'pi')
        ]
        self.shift_panel = tk.Frame(parent, bg="#d0d7de")
        for idx, (label, insert_value) in enumerate(shift_options):
            row = idx // 4
            column = idx % 4
            btn = tk.Button(
                self.shift_panel,
                text=label,
                font=('arial', 12),
                bg="#c1d3e0",
                activebackground="#99b6d6",
                relief='raised',
                borderwidth=2,
                command=lambda v=insert_value: self.insert_shift_value(v)
            )
            btn.grid(row=row, column=column, sticky="nsew", padx=2, pady=2)
        total_rows = max(1, (len(shift_options) + 3) // 4)
        for r in range(total_rows):
            self.shift_panel.grid_rowconfigure(r, weight=1)
        for c in range(4):
            self.shift_panel.grid_columnconfigure(c, weight=1)
        self.shift_panel.grid(row=7, column=0, columnspan=4, sticky="nsew", padx=2, pady=(0, 4))
        self.shift_panel.grid_remove()

    def toggle_shift(self):
        self.shift_active = not self.shift_active
        if self.shift_active:
            self.shift_panel.grid()
            if self.shift_button is not None:
                self.shift_button.config(bg="#4CAF50", fg="white")
        else:
            self.shift_panel.grid_remove()
            if self.shift_button is not None:
                self.shift_button.config(bg="#607D8B", fg="white")

    def insert_shift_value(self, value):
        self.expression += str(value)
        self.input_text.set(self.expression)
        if self.shift_active:
            self.toggle_shift()

    def _preprocess_for_sympy(self, expr_str):
        processed = expr_str.replace('^', '**')
        return processed

    def evaluate_expression(self):
        if not self.expression:
            return
        try:
            if 'x' in self.expression:
                if '=' not in self.expression:
                    self.expression += '='
                    self.input_text.set(self.expression) # Update display
                else:
                    messagebox.showinfo("Info", "Equation format started. Complete the right-hand side or use 'Solve x'.")
                return 

            processed_expr_str = self._preprocess_for_sympy(self.expression)
            expr_obj = sympy.parse_expr(processed_expr_str, local_dict=SYMPY_GLOBALS, transformations='all')
            result = expr_obj.evalf()

            if result == sympy.zoo: 
                self.show_error("Result is complex infinity (e.g., log of negative).")
                return
            if result == sympy.oo or result == -sympy.oo: 
                self.show_error("Result is infinity (e.g., division by zero).")
                return

            result_str = str(result)
            self.input_text.set(result_str)
            self.expression = result_str 

        except (sympy.SympifyError, SyntaxError) as e:
            self.show_error(f"Syntax Error: {e}")
        except TypeError as e:
            self.show_error(f"Type Error (e.g. log(-1)): {e}")
        except ZeroDivisionError: 
            self.show_error("Division by zero")
        except Exception as e:
            self.show_error(f"Error: {e}")

    def solve_for_x(self):
        if not self.expression:
            return
        if 'x' not in self.expression: 
            messagebox.showinfo("Info", "No variable 'x' found in the expression to solve for.")
            return

        current_expr_for_solve = self.expression
        if '=' not in current_expr_for_solve:
            current_expr_for_solve += '=0'
            # self.input_text.set(current_expr_for_solve) # Optionally update display here
            # self.expression = current_expr_for_solve # And internal state

        try:
            processed_expr_str = self._preprocess_for_sympy(current_expr_for_solve)
            
            parts = processed_expr_str.split('=', 1)
            if len(parts) != 2:
                self.show_error("Invalid equation format. Expected 'LHS = RHS'.")
                return

            lhs_str = parts[0].strip()
            rhs_str = parts[1].strip()

            if not lhs_str:
                self.show_error("Invalid equation: Missing left-hand side.")
                return
            if not rhs_str: 
                 rhs_str = "0"

            local_parsing_dict = SYMPY_GLOBALS.copy()

            try:
                lhs_expr = sympy.parse_expr(lhs_str, local_dict=local_parsing_dict, transformations='all')
                rhs_expr = sympy.parse_expr(rhs_str, local_dict=local_parsing_dict, transformations='all')
            except Exception as parse_e:
                self.show_error(f"Error parsing equation sides: {parse_e}\nInput was: LHS='{lhs_str}', RHS='{rhs_str}'\nEnsure expressions are valid.")
                return

            if x_sym not in lhs_expr.free_symbols and x_sym not in rhs_expr.free_symbols:
                 if 'x' in lhs_str or 'x' in rhs_str:
                    self.show_error(f"Variable 'x' was typed but not recognized as a solvable symbol in the equation parts. Check syntax around 'x'.")
                 else:
                    messagebox.showinfo("Info", "No variable 'x' found in the equation parts to solve for.")
                 return

            equation = sympy.Eq(lhs_expr, rhs_expr)
            solutions = sympy.solve(equation, x_sym)

            result_display_string = "" 

            if not solutions:
                is_identity_or_contradiction = False
                try:
                    simplified_eq = sympy.simplify(equation) # simplify(lhs - rhs) is another way
                    if simplified_eq is sympy.true: # More robust check
                        result_display_string = "True for all x"
                        is_identity_or_contradiction = True
                    elif simplified_eq is sympy.false: # More robust check
                        result_display_string = "No solution (Contradiction)"
                        is_identity_or_contradiction = True
                except Exception:
                    pass 
                
                if not is_identity_or_contradiction:
                    result_display_string = "No specific solution found for x"
            else:
                solution_parts = []
                for s in solutions:
                    if isinstance(s, dict):
                        solution_parts.append(str(s))
                        continue
                    try:
                        if hasattr(s, 'evalf'):
                            num_s = s.evalf(5)
                            solution_parts.append(str(num_s))
                        elif isinstance(s, (sympy.Integer, sympy.Float, sympy.Rational)):
                            solution_parts.append(str(s))
                        else:
                            solution_parts.append(str(s))
                    except Exception:
                        solution_parts.append(str(s))
                result_display_string = f"x = {', '.join(solution_parts)}"
            
            self.input_text.set(result_display_string)
            self.expression = result_display_string # CRITICAL: Update self.expression

        except (sympy.SympifyError, SyntaxError) as e:
            self.show_error(f"Equation Syntax Error: {e}")
        except NotImplementedError:
            self.show_error("Solver cannot handle this type of equation.")
        except Exception as e:
            self.show_error(f"Solving Error: {e}\nEquation was: {current_expr_for_solve}")


    def show_error(self, message):
        messagebox.showerror("Calculator Error", message)
        # Potentially clear expression on error, or leave it for user to correct
        # self.expression = "" 
        # self.input_text.set(self.expression)

# --- Main execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedCalculator(root)
    root.mainloop()